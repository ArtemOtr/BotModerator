from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
from datetime import datetime, timedelta
import logging
import traceback
import torch
import db
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.dispatcher.filters.state import State, StatesGroup
from joblib import load
from data_preproc import count_vectorizer, count_vectorizer2, tf_idf_vectorizer
from rnn_model import custom_tokenizer, text_to_indices, vocab_1, vocab_2, max_len_1, max_len_2, LSTM


def predict_lstm_text(text, type):
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    data = {'toxic':{'vocab':vocab_1, 'max_len':max_len_1, 'model' : lstm_toxic} , 'spam': {'vocab':vocab_2, 'max_len': max_len_2, 'model' : lstm_toxic} }
    text_indices = text_to_indices(text, data[type]['vocab'], data[type]['max_len'])
    tensor = torch.tensor([text_indices], device=device)
    hidden_size = 64
    num_layers = 3

    hidden = torch.zeros(num_layers, tensor.size(0), hidden_size, device=device)
    memory = torch.zeros(num_layers, tensor.size(0), hidden_size, device=device)

    with torch.no_grad():
        pred, _, _ = data[type]['model'](tensor, hidden, memory)
        predicted_class = pred[:, -1, :].argmax(dim=1).item()
    return predicted_class


def predict_text(text, model, vectorizer):
    prediction = model.predict(vectorizer.transform([text]))
    return prediction[0]


API_TOKEN = "YOUR_TOKEN"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
database = db.Database()


#модели
svm_toxic = load('svm_toxic.pkl')
svm_spam = load('svm_spam.pkl')
log_reg_toxic = load('logistic_regression_toxic.pkl')
log_reg_spam = load('logistic_regression_spam.pkl')
mnbc_toxic = load('naive_bayes_toxic.pkl')
mnbc_spam = load('naive_bayes_spam.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
num_layers = 3

print('Перед инициализацией')

lstm_toxic = LSTM(num_emb=len(vocab_1), output_size=2,
                         num_layers=num_layers, hidden_size=hidden_size).to(device)
lstm_spam = LSTM(num_emb=len(vocab_2), output_size=2,
                         num_layers=num_layers, hidden_size=hidden_size).to(device)

print('Перед загрузкой весов')

lstm_toxic.load_state_dict(torch.load('C:/Users/nenad/TgBot/lstm_model_toxic.pth', weights_only=True))
lstm_spam.load_state_dict(torch.load('C:/Users/nenad/TgBot/lstm_model_spam.pth', weights_only=True))
print('После загрузки весов')
lstm_toxic.to(device)
lstm_spam.to(device)
lstm_toxic.eval()
lstm_spam.eval()
print('модели загружены')

class Form(StatesGroup):
    toxic_punishment = State()
    spam_punishment = State()
async def is_muted_query():
    while True:
        muted_list = database.get_list_of_muted_users()
        for user in muted_list:
            if datetime.now() > user[-1]:
                username = db.process_cmd('''SELECT username FROM users WHERE user_id = %s''', [user[0]])[0][0]
                tg_chat_id = db.process_cmd('''SELECT tg_chat_id FROM chats WHERE chat_id = %s''', [user[1]])[0][0]
                database.unmute_user(username=username, tg_chat_id=tg_chat_id)
        await asyncio.sleep(20)





@dp.message_handler(content_types=types.ContentTypes.NEW_CHAT_MEMBERS)
async def bot_added_to_chat(message: types.Message):
    ADMIN_USER_ID = message.from_user.id
    chat_id = message.chat.id

    #добавление записи о новом чате в БД
    database.new_chat(ADMIN_USER_ID, chat_id)

    #дальше просто из во всех функциях вытаскиваем  chat_id и обращаемся
    for member in message.new_chat_members:
        if member.id == bot.id:
            # Уведомляем администратора, если известен его user_id
            if ADMIN_USER_ID:
                try:
                    print(ADMIN_USER_ID)
                    await bot.send_message(
                        ADMIN_USER_ID,
                        f"Бот был добавлен в группу '{message.chat.title}'. Пожалуйста, предоставьте мне права администратора."
                    )

                    keyboard_toxic = InlineKeyboardMarkup(row_width=4)
                    svm_btn_toxic = InlineKeyboardButton("SVM", callback_data=f"svm_toxic_{chat_id}")
                    logreg_btn_toxic = InlineKeyboardButton("LogisticRegression", callback_data=f"logreg_toxic_{chat_id}")
                    lstm_btn_toxic = InlineKeyboardButton("LSTM", callback_data=f"lstm_toxic_{chat_id}")
                    mnbc_btn_toxic = InlineKeyboardButton("MNBC", callback_data=f"mnbc_toxic_{chat_id}")
                    keyboard_toxic.add(svm_btn_toxic, logreg_btn_toxic, lstm_btn_toxic, mnbc_btn_toxic)

                    await bot.send_message(
                        ADMIN_USER_ID,
                        f"Выберите модель для классификации токсичных сообщений.", reply_markup=keyboard_toxic
                    )



                except:
                    print("Бот не может отправить сообщение админу. Возможно, он заблокировал бота.")
        else:
            if not database.is_banned(member.username, chat_id):
                database.add_user(member.username, chat_id)
                await message.reply(f"Привет, {member.first_name}!")
                print(member)
            else:
                await message.reply(f"Пользователь {member.first_name} забанен и будет исключен из чата до разбана!")
                await bot.kick_chat_member(chat_id=message.chat.id, user_id=member.id)
            break
    return ADMIN_USER_ID

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    if message.chat.type == "private":
        await message.reply(
            "Привет! Добавьте меня в чат и дайте права администратора. После этого я начну работу!"
        )
    else:
        await message.reply("Я уже работаю в группе. Готов помочь с модерацией!")

@dp.message_handler(commands=['help'], commands_prefix = '/')
async def help_command(message: types.Message):
    if message.chat.type in ['private']:
        with open("help_command.txt", "r", encoding="utf-8") as file:
            help = file.read()
        await message.reply(help, parse_mode="HTML")
        return


@dp.message_handler(commands=['set_classification_parametrs'], commands_prefix='/')
async def set_classification_parametrs(message: types.Message):
    if message.chat.type in ['group', 'supergroup', 'channel']:
        user_id = message.from_user.id
        chat_id = message.chat.id
        keyboard_toxic = InlineKeyboardMarkup(row_width=4)
        svm_btn_toxic = InlineKeyboardButton("SVM", callback_data=f"svm_toxic_{chat_id}")
        logreg_btn_toxic = InlineKeyboardButton("LogisticRegression", callback_data=f"logreg_toxic_{chat_id}")
        lstm_btn_toxic = InlineKeyboardButton("LSTM", callback_data=f"lstm_toxic_{chat_id}")
        mnbc_btn_toxic = InlineKeyboardButton("MNBC", callback_data=f"mnbc_toxic_{chat_id}")
        keyboard_toxic.add(svm_btn_toxic, logreg_btn_toxic, lstm_btn_toxic, mnbc_btn_toxic)

        await bot.send_message(
            user_id,
            f"Выберите модель для классификации токсичных сообщений.", reply_markup=keyboard_toxic
        )



@dp.callback_query_handler(lambda c: c.data and c.data.split('_')[1] == 'toxic')
async def handle_toxic_query(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    chat_id = callback_query.message.chat.id
    member = await bot.get_chat_member(chat_id, user_id)

    if member.status not in ["administrator", "creator"] and callback_query.message.chat.type != 'private':
        await callback_query.answer("Только администратор может выполнять эту операцию.", show_alert=True)
        return


    chat_id = int(callback_query.data.split('_')[2])
    user_id = database.get_admin_id(chat_id)
    database.add_model_to_chat(type = 'toxic', model = callback_query.data.split('_')[0], tg_chat_id=chat_id )


    keyboard_spam = InlineKeyboardMarkup(row_width=4)
    svm_btn_spam = InlineKeyboardButton("SVM", callback_data=f"svm_spam_{chat_id}")
    logreg_btn_spam = InlineKeyboardButton("LogisticRegression", callback_data=f"logreg_spam_{chat_id}")
    lstm_btn_spam = InlineKeyboardButton("LSTM", callback_data=f"lstm_spam_{chat_id}")
    mnbc_btn_spam = InlineKeyboardButton("MNBC", callback_data=f"mnbc_spam_{chat_id}")
    keyboard_spam.add(svm_btn_spam, logreg_btn_spam, lstm_btn_spam, mnbc_btn_spam)

    await bot.send_message(
        user_id,
        f"Выберите модель для классификации спама.", reply_markup=keyboard_spam
    )


@dp.callback_query_handler(lambda c: c.data and c.data.split('_')[1] == 'spam')
async def handle_spam_query(callback_query: CallbackQuery):
    chat_id = int(callback_query.data.split('_')[2])
    user_id = database.get_admin_id(chat_id)


    database.add_model_to_chat(type='spam', model=callback_query.data.split('_')[0], tg_chat_id=chat_id)

    await bot.send_message(
        user_id,
        '''Выберите наказание в случае автоматической детекциии токсичного сообщения.
         \n Введите сообщение в формате: ban или mute 1d 1h 1m 1s''',
    )
    await Form.toxic_punishment.set()

@dp.message_handler(state=Form.toxic_punishment)
async def add_toxic_punishment(message: types.Message):
    user_id = message.from_user.id
    chat_id = int(db.process_cmd('SELECT tg_chat_id FROM chats WHERE admin_id = %s', values = [str(user_id)])[0][0])
    msg = message.text.split()
    duration = None
    if len(msg) > 1:
        duration = ' '.join(msg[1:])
    database.add_punishment(type = 'toxic', punishment=msg[0], tg_chat_id=chat_id, duration=duration)

    await bot.send_message(
        user_id,
        '''Выберите наказание в случае автоматической детекциии спама.
         \n Введите сообщение в формате: ban или mute 1d 1h 1m 1s''',
    )
    await Form.spam_punishment.set()

@dp.message_handler(state=Form.spam_punishment)
async  def add_spam_punishment(message: types.message):

    user_id = message.from_user.id

    chat_id = int(db.process_cmd('SELECT tg_chat_id FROM chats WHERE admin_id = %s', values=[str(user_id)])[0][0])
    msg = message.text.split()
    if not (msg[0] in ['ban', 'mute']):
        await bot.send_message(
            user_id,
            '''Введите команду в формате: ban или mute 1d 1h 1m 1s''', )

        return
    duration = None
    if len(msg) > 1:
        duration = ' '.join(msg[1:])
    database.add_punishment(type='spam', punishment=msg[0], tg_chat_id=chat_id, duration=duration)
    await bot.send_message(
        user_id,
        '''Настройка бота завершена''',)




@dp.chat_member_handler()
async def check_admin_status(chat_id: int):
    try:
        member = await bot.get_chat_member(chat_id, bot.id)
        return member.status in ["administrator", "creator"]
    except Exception:
        return False


@dp.message_handler(commands=['ban'], commands_prefix='/')
async def ban_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return
    chat_id = message.chat.id
    command_parts = message.text.split(maxsplit=1)
    if not message.reply_to_message:
        await message.reply(
            "Эта команда должна быть ответом на сообщение пользователя, которого вы хотите заблокировать.")
        return

    user_to_ban = message.reply_to_message.from_user

    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return

    reason = command_parts[1] if len(command_parts) > 1 else 'Не указано'

    await bot.kick_chat_member(chat_id=message.chat.id, user_id=user_to_ban.id)

    database.ban_user(user_to_ban.username, chat_id, reason = reason)
    await message.reply(f"Пользователь {user_to_ban.first_name} был забанен.")



@dp.message_handler(commands=['unban'], commands_prefix='/')
async def unban_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return
    chat_id = message.chat.id
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return
    if len(message.text.split()) < 2:
        await message.reply("Используйте команду в формате /unban @username")
        return

    # Извлекаем имя пользователя
    username = message.text.split()[1]
    print(username)
    if username[0] != "@":
        await message.reply("Имя пользователя должно начинаться с '@'. Например, /unban @username")
        return

    username = username[1:]

    try:

        if not database.get_id_by_name(username, chat_id):
            await message.reply(f"Пользователь с именем @{username} не найден в этом чате.")
            return

        await message.reply(f"Пользователь @{username} был разблокирован.")
        database.unban_user(username, chat_id)
    except Exception as e:
        await message.reply("Произошла ошибка при попытке разблокировать пользователя.")
        print(f"Ошибка: {e}")

@dp.message_handler(commands=['mute'], commands_prefix='/')
async def mute_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return
    chat_id = message.chat.id
    command_parts = message.text.split(' ', 1)[1]
    command_parts = command_parts.split()
    time_dict = {'d':0, 'h': 0, 'm':0, 's': 0}
    reason = ''
    for i in range(len(command_parts)):
        if command_parts[i][-1] in time_dict.keys() and command_parts[i][0].isdigit():
            try:
                time_dict[command_parts[i][-1]] = int(command_parts[i][:-1])
            except:
                await message.reply(
                    "Эта команда должна быть ответом на сообщение пользователя, которого вы хотите замутить. В формате /mute 1h 1m 1s *причина*")
                return
        else:
            reason += command_parts[i]
            reason += ' '
    if not message.reply_to_message:
        await message.reply(
            "Эта команда должна быть ответом на сообщение пользователя, которого вы хотите замутить. В формате /mute 1h 1m 1s *причина*")
        return

    user_to_mute = message.reply_to_message.from_user

    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return


    database.mute_user(user_to_mute.username, time_dict=time_dict, tg_chat_id=chat_id, reason = reason)
    await message.reply(f"Пользователь {user_to_mute.first_name} был замучен.")


@dp.message_handler(commands=['unmute'], commands_prefix='/')
async def unmute_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return

    chat_id = message.chat.id

    if len(message.text.split()) < 2:
        await message.reply("Используйте команду в формате /unmute @username")
        return

    # Извлекаем имя пользователя
    username = message.text.split()[1]
    if username[0] != "@":
        await message.reply("Имя пользователя должно начинаться с '@'. Например, /unmute @username")
        return

    username = username[1:]  # Убираем '@' для поиска

    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return

    if not database.get_id_by_name(username, chat_id):
        await message.reply(f"Пользователь с именем @{username} не найден в этом чате.")
        return
    if not database.is_muted(username, chat_id):
        await message.reply(f"Пользователь с именем @{username} не находится в муте.")
        return

    # Разбан пользователя
    database.unmute_user(username, chat_id)
    await message.reply(f"Пользователь @{username} был размучен.")


@dp.message_handler(commands=['set_warn_limit'], commands_prefix='/')
async def warn_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return

    chat_id = message.chat.id
    warn_limit = int(message.text.split(' ', 1)[1])


    try:
        database.set_warn_limit(tg_chat_id=chat_id,warn_limit=warn_limit )
        await message.reply(f"Лимит варнов успешно установлен. Текущий лимит {warn_limit}")
    except:
        await message.reply(f"Ошибка. Команда должна быть в формате /set_warn_limit *число*")
        return


@dp.message_handler(commands=['warn'], commands_prefix = '/')
async def warn_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return
    chat_id = message.chat.id
    command_parts = message.text.split(maxsplit=1)
    if not message.reply_to_message:
        await message.reply(
            "Эта команда должна быть ответом на сообщение пользователя, которому вы хотите дать варн.")
        return

    user_to_warn = message.reply_to_message.from_user

    # Проверяем, имеет ли пользователь права администратора
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return

    reason = command_parts[1] if len(command_parts) > 1 else 'Не указано'


    database.warn_user(user_to_warn.username, chat_id, reason=reason)

    warn_count = database.get_warn_count(user_to_warn.username, chat_id)
    warn_limit = database.get_warn_limit(chat_id)

    await message.reply(f"Пользователь {user_to_warn.first_name} было выдано предупреждение. Это {warn_count} из {warn_limit} предупреждение.")
    if isinstance(warn_limit, int) and warn_limit <= warn_count:
        await bot.kick_chat_member(chat_id=chat_id, user_id=user_to_warn.id)
        database.ban_user(user_to_warn.username, chat_id, reason=reason)
        await message.reply(f"Пользователь {user_to_warn.first_name} был забанен.")
        database.delete_all_warns(username = user_to_warn.username, tg_chat_id=chat_id)

@dp.message_handler(commands=['unwarn'], commands_prefix = '/')
async def unwarn_user(message: types.Message):
    chat_id = message.chat.id
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return

    if len(message.text.split()) < 2:
        await message.reply("Используйте команду в формате /unwarn @username *mute_id*(необязательный аргумент)")
        return

    # Извлекаем имя пользователя
    username = message.text.split()[1]
    warn_id = None
    if len(message.text.split()) > 2:
        warn_id  =  message.text.split()[2]
    if username[0] != "@":
        await message.reply("Имя пользователя должно начинаться с '@'. Например, /unwarn @username *mute_id*(необязательный аргумент)")
        return

    username = username[1:]  # Убираем '@' для поиска

    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return

    if not database.get_id_by_name(username, chat_id):
        await message.reply(f"Пользователь с именем @{username} не найден в этом чате.")
        return

    database.unwarn_user(tg_chat_id= chat_id, warn_id=warn_id, username=username)

    await message.reply(f"C пользователя @{username} был снят варн.")

@dp.message_handler(commands=['get_info_about_user'], commands_prefix = '/')
async def get_info_about_user(message: types.Message):
    if message.chat.type not in ["group", "supergroup", 'channel']:
        await message.reply("Эта команда доступна только в групповых чатах.")
        return

    chat_id = message.chat.id
    if len(message.text.split()) < 2:
        await message.reply("Используйте команду в формате /get_info_about_user @username")
        return
    username = message.text.split()[1]
    if username[0] != "@":
        await message.reply("Имя пользователя должно начинаться с '@'. Например, /unwarn @username *mute_id*(необязательный аргумент)")
        return

    username = username[1:]
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ["administrator", "creator"]:
        await message.reply("У вас нет прав для использования этой команды.")
        return
    if not database.get_id_by_name(username, chat_id):
        await message.reply(f"Пользователь с именем @{username} не найден в этом чате.")
        return
    user_info = database.get_information_about_user(username=username, tg_chat_id=chat_id)
    reply = ''
    for key, value in user_info.items():
        if not isinstance(value, dict):
            reply += f'{key}: {value} \n'
        else:
            reply += f'{key}: \n'
            for keyj, valuej in value.items():
                reply += f'{keyj}: {valuej} \n'

    await message.reply(reply)



#обработка сообщений для бана и тд тож здесь
@dp.message_handler()
async def users_message_handler(message: types.Message):
    if message.chat.type in ['group', 'supergroup', 'channel']:
        print(message)
        chat_id = message.chat.id
        username = message.from_user.username
        database.add_user(username, chat_id)
        if database.is_muted(username, chat_id):
            await message.delete()
            return
        models = {
                'svm': {'toxic':svm_toxic, 'spam': svm_spam},
                'lstm' : {'toxic':lstm_toxic, 'spam': lstm_spam},
                'mnbc' : {'toxic':mnbc_toxic, 'spam':mnbc_spam},
                'log_reg' : {'toxic': log_reg_toxic, 'spam': log_reg_spam}
        }
        toxic_model = models[database.get_model(tg_chat_id=chat_id,type='toxic')]['toxic']
        spam_model = models[database.get_model(tg_chat_id=chat_id,type='spam')]['spam']
        toxic_mute = db.process_cmd('SELECT toxic_mute FROM chats WHERE tg_chat_id = %s', [str(chat_id)])[0][0]
        if toxic_model == lstm_toxic:
            toxic_flag = predict_lstm_text(text=message.text, type='toxic')
        else:
            toxic_flag = predict_text(text=message.text, model=toxic_model, vectorizer=count_vectorizer)
        if toxic_model == lstm_toxic:
            spam_flag = predict_lstm_text(text=message.text, type='spam')
        else:
            spam_vectorizer = tf_idf_vectorizer if spam_model == 'svm_spam' else count_vectorizer2
            spam_flag = predict_text(text=message.text, model=spam_model, vectorizer=spam_vectorizer)
        print(f'is toxic: {toxic_flag}, is spam: {spam_flag}')
        if toxic_flag:
            if db.process_cmd('SELECT toxic_ban FROM chats WHERE tg_chat_id = %s', [str(chat_id)])[0][0]:
                await message.reply(f"Пользователь @{username} забанен за токсичное сообщение")
                await message.delete()
                database.ban_user(username=username,tg_chat_id=chat_id, reason = 'Токсичное сообщение')
                await bot.kick_chat_member(chat_id=chat_id, user_id=message.from_user.id)
            elif toxic_mute:
                await message.reply(f"Пользователь @{username} замучен за токсичное сообщение")
                await message.delete()
                timing = {'d': 0,'h':0, 'm':0, 's': 0}
                toxic_mute = toxic_mute.split('_')
                for i in range(len(toxic_mute)):
                    timing[toxic_mute[i][-1]] = int(toxic_mute[i][0])
                database.mute_user(username=username, time_dict=timing, tg_chat_id=chat_id, reason='Токсичное сообщение' )
            else:
                print(f'Токичное сообщение от @{username}. Наказаний не предусмотрено')
                return

        if spam_flag:
            if db.process_cmd('SELECT toxic_ban FROM chats WHERE tg_chat_id = %s', [str(chat_id)])[0][0]:
                await message.reply(f"Пользователь @{username} забанен за токсичное сообщение")
                await message.delete()
                database.ban_user(username=username, tg_chat_id=chat_id, reason='Спам')
                await bot.kick_chat_member(chat_id=chat_id, user_id=message.from_user.id)
            elif toxic_mute:
                await message.reply(f"Пользователь @{username} замучен за спам")
                await message.delete()
                timing = {'d': 0, 'h': 0, 'm': 0, 's': 0}
                toxic_mute = toxic_mute.split('_')
                for i in range(len(toxic_mute)):
                    timing[toxic_mute[i][-1]] = int(toxic_mute[i][0])
                database.mute_user(username=username, time_dict=timing, tg_chat_id=chat_id, reason='Спам')
            else:
                print(f'Спамерское сообщение от @{username}. Наказаний не предусмотрено')
                return




from aiogram.types import ContentType

@dp.message_handler(content_types=[
    ContentType.VIDEO_NOTE,  # Кружки
    ContentType.PHOTO,       # Фотографии
    ContentType.DOCUMENT,    # Файлы
    ContentType.ANIMATION,   # Гифки
    ContentType.VOICE,        # Голосовые
    ContentType.STICKER
])
async def handle_all_media(message: types.Message):

    chat_id = message.chat.id
    username = message.from_user.username
    database.add_user(username, chat_id)
    if database.is_muted(username, chat_id):
        await message.delete()
        return


async def on_startup(dispatcher):
    """Функция, которая выполняется при старте бота."""
    logging.info("Бот запускается...")
    asyncio.create_task(is_muted_query())

@dp.errors_handler()
async def global_error_handler(update, exception):
    logging.error("Ошибка в функции:")
    logging.error(traceback.format_exc())  # Записывает полный стек вызовов
    return True


# Запуск бота
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)

