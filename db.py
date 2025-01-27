import psycopg2
from datetime import datetime, timedelta
import pytz

DB_HOST = ""
DB_NAME = ""
DB_USER = ""
DB_PASS = ""


def process_cmd(cmd, values=None):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        cur.execute(cmd, values if values else [])
        try:
            rows = cur.fetchall()
        except psycopg2.ProgrammingError:
            conn.commit()
            rows = []
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error processing command: {e}")
        return []

class Database():
#        def __init__(self, chat_link, admin, timezone = 'Europe/Moscow'):
#        self.timezone = pytz.timezone(timezone)
#        self.chat_link = chat_link
#        self.admin = admin
#        process_cmd('''INSERT INTO chats (chat_link, admin_id)
#                VALUES (%s, %s);''', values=[self.chat_link, self.admin])
#        self.chat_id = process_cmd('''SELECT FROM chats chat_id WHERE chat_link = %s AND admin_id = %s''', values=[self.chat_link, self.admin])

    def get_list_of_muted_users(self):
        return process_cmd('''SELECT user_id, chat_id, muted_until FROM mutes''')

    def new_chat(self, admin, tg_chat_id):
        if not process_cmd('''SELECT * FROM chats WHERE tg_chat_id = %s ''', [str(tg_chat_id) ]):
            print('slkfjsdf')
            process_cmd('''INSERT INTO chats (tg_chat_id, admin_id)
                            VALUES (%s, %s);''', values=[tg_chat_id, admin])

    def get_admin_id(self, tg_chat_id):
        return int(process_cmd('''SELECT admin_id FROM chats WHERE tg_chat_id = %s''', [str(tg_chat_id)])[0][0])


    def get_chat_id_by_tg_chat_id(self, tg_chat_id):
        return process_cmd('''SELECT chat_id FROM chats WHERE tg_chat_id = %s''', [str(tg_chat_id)])[0][0]

    def get_id_by_name(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        q = '''SELECT user_id FROM users
        WHERE username = %s AND chat_id = %s;'''
        res = process_cmd(q, [username, chat_id])[0][0]
        return res


    def add_model_to_chat(self, type, model, tg_chat_id):
        types = {'toxic': 'toxic_model' ,
                   'spam':  'spam_model'   }
        process_cmd(f'UPDATE chats SET {types[type]} = %s WHERE tg_chat_id = %s', [model, str(tg_chat_id)])
    def add_punishment(self, type, punishment, tg_chat_id, duration = None):
        change = duration if duration else True
        process_cmd(f'UPDATE chats SET {type}_mute = NULL, {type}_ban = FALSE WHERE tg_chat_id = %s', values= [str(tg_chat_id)] )
        process_cmd(f'UPDATE chats SET {type}_{punishment} = %s WHERE tg_chat_id = %s', values= [change, str(tg_chat_id)] )
        process_cmd('')

    #добавление нового пользователя
    def add_user(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        if not process_cmd('''SELECT * FROM users WHERE username = %s AND chat_id = %s''', [username, chat_id]):
            q = '''INSERT INTO users (username, chat_id)
                    VALUES (%s, %s);'''
            process_cmd(q, [username, chat_id])

    def get_model(self, tg_chat_id, type):
        return process_cmd(f'''SELECT {type}_model FROM chats WHERE tg_chat_id = %s''', values = [str(tg_chat_id)])[0][0]
    # метод для админа получения информации о пользователе по username
    def get_information_about_user(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        q = '''SELECT * FROM users
            WHERE username = %s AND chat_id = %s;'''
        info = process_cmd(q, values=[username, chat_id])
        info = info[0]
        if not info:
            return None
        info = {'user-id' : info[0], 'username' : info[1], 'Забанен' : info[2], 'Замучен' : info[3], 'Количество варнов' : info[4]}
        if info['Забанен']:
            info['Причина бана'], info['Дата бана'] = process_cmd('''SELECT reason, banned_at FROM bans WHERE  user_id = %s AND chat_id = %s;''', [self.get_id_by_name(username, tg_chat_id), chat_id])[0]
            info['Дата бана'] = info['Дата бана'].strftime(f'%Y-%m-%d %H:%M:%S')
        if info['Замучен']:
            info['Причина мута'], info['Дата мута'], info['Замучен до'] = process_cmd('''SELECT reason, muted_at, muted_until FROM mutes WHERE  user_id = %s AND chat_id = %s;''', [self.get_id_by_name(username, tg_chat_id), chat_id])[0]
            info['Дата мута'] = info['Дата мута'].strftime(f'%Y-%m-%d %H:%M:%S')
            info['Замучен до'] = info['Замучен до'].strftime(f'%Y-%m-%d %H:%M:%S')
        if info['Количество варнов'] != 0:
            warns = process_cmd('''SELECT reason, warned_at, warn_id FROM warns WHERE user_id = %s AND chat_id = %s;''', [self.get_id_by_name(username, tg_chat_id), chat_id])
            info['Инофрмация о варнах'] = {}
            for i in range(len(warns)):
                info['Инофрмация о варнах'][i + 1] = {}
                info['Инофрмация о варнах'][i + 1]['Причина варна'] = warns[i][0]
                info['Инофрмация о варнах'][i + 1]['Дата варна'] = warns[i][1].strftime("%d.%m.%Y %H:%M:%S")
                info['Инофрмация о варнах'][i + 1]['warn_id'] = warns[i][2]

        return info
    # метод чтобы забанить пользователя
    def ban_user(self, username, tg_chat_id, reason = None):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        if reason:
            q = '''INSERT INTO bans (user_id, reason, chat_id)
            VALUES (%s, %s, %s)'''
            process_cmd(q, values=[self.get_id_by_name(username, tg_chat_id), reason, chat_id])
        else:
            q = '''INSERT INTO bans (user_id, chat_id)
            VALUES (%s, %s)'''
            process_cmd(q, values=[self.get_id_by_name(username, tg_chat_id), chat_id])
        process_cmd('''UPDATE users SET is_banned = TRUE WHERE username = %s AND chat_id = %s''', [username, chat_id])

    # метод чтобы замутить пользователя
    # строка времени имеет вид 6d 5h 4m 3s
    def mute_user(self, username, time_dict, tg_chat_id, reason = None):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        process_cmd('''UPDATE users SET is_muted = TRUE WHERE username = %s  AND chat_id = %s''', [username, chat_id])
        curr_time = datetime.now()
        time_delta = timedelta(days=time_dict['d'], hours=time_dict['h'], minutes=time_dict['m'], seconds=time_dict['s'])
        end_time = curr_time + time_delta
        mute_dur_sec = time_dict['d'] * 86400 + time_dict['h'] * 3600 + time_dict['m'] * 60 + time_dict['s']
        if reason:
            q = '''INSERT INTO mutes (user_id, reason, mute_dur_sec, muted_until, chat_id)
            VALUES (%s, %s, %s, %s, %s)'''
            process_cmd(q, values = [self.get_id_by_name(username, tg_chat_id), reason, mute_dur_sec, end_time.strftime(f'%Y-%m-%d %H:%M:%S'), chat_id])
        else:
            q = '''INSERT INTO mutes (user_id, mute_dur_sec, muted_until, chat_id)
            VALUES (%s, %s, %s, %s)'''
            process_cmd(q, values = [self.get_id_by_name(username, tg_chat_id), mute_dur_sec, end_time.strftime(f'%Y-%m-%d %H:%M:%S'), chat_id])
        

    # метод чтобы заварнить пользователя
    def warn_user(self, username, tg_chat_id, reason = None):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        process_cmd('''UPDATE users SET warn_count = warn_count + 1 WHERE username = %s AND chat_id = %s''', [username, chat_id])
        if reason:
            q = '''INSERT INTO warns (user_id, reason, chat_id)
            VALUES (%s, %s, %s)'''
            process_cmd(q, values=[self.get_id_by_name(username, tg_chat_id), reason, chat_id])
        else:
            q = '''INSERT INTO warns (user_id, chat_id)
            VALUES (%s, %s)'''
            process_cmd(q, values=[self.get_id_by_name(username, tg_chat_id), chat_id])


    # проверка на бан
    def is_banned(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        return process_cmd('''SELECT is_banned FROM users WHERE user_id = %s AND chat_id = %s''', [self.get_id_by_name(username, tg_chat_id), chat_id])[0][0]
    # проверка на мут
    def is_muted(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        user_id = self.get_id_by_name(username, tg_chat_id)
        if process_cmd('''SELECT is_muted FROM users WHERE user_id = %s AND chat_id = %s''', [self.get_id_by_name(username, tg_chat_id), chat_id], )[0][0]:
            return process_cmd('''SELECT muted_until FROM mutes WHERE user_id = %s AND chat_id = %s''', [user_id, chat_id])[0][0]
        return False
    # проверка на варн
    def get_warn_count(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        return process_cmd('''SELECT warn_count FROM users WHERE user_id = %s AND chat_id = %s''', [self.get_id_by_name(username, tg_chat_id), chat_id])[0][0]

    # метод разбана
    def unban_user(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        process_cmd('''DELETE FROM bans WHERE user_id = %s AND chat_id = %s''',  values=[self.get_id_by_name(username, tg_chat_id), chat_id])
        process_cmd('''UPDATE users SET is_banned = FALSE WHERE username = %s AND chat_id = %s''', [username, chat_id])
    # метод размута
    def unmute_user(self,  username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        process_cmd('''DELETE FROM mutes WHERE user_id = %s AND chat_id = %s''',  values=[self.get_id_by_name(username, tg_chat_id), chat_id])
        process_cmd('''UPDATE users SET is_muted = FALSE WHERE username = %s AND chat_id = %s''', [username, chat_id])
    # метод снятия варна
    def unwarn_user(self, tg_chat_id, username, warn_id = None):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        if warn_id:
            process_cmd('''DELETE FROM warns WHERE warn_id = %s  AND chat_id = %s''', values = [warn_id, chat_id])
        else:
            process_cmd('''DELETE FROM warns WHERE warn_id = (SELECT warn_id FROM warns WHERE
             chat_id = %s ORDER BY warn_id ASC LIMIT 1 );''', values = [chat_id])
        warn_count = process_cmd('''SELECT warn_count FROM users WHERE username = %s AND chat_id = %s''', [username, chat_id])[0][0]
        process_cmd('''UPDATE users SET warn_count = %s WHERE username = %s AND chat_id = %s''', [warn_count - 1, username, chat_id])


    def set_warn_limit(self, tg_chat_id, warn_limit):
        process_cmd('''UPDATE chats SET warn_limit = %s WHERE tg_chat_id = %s''', values=[warn_limit, str(tg_chat_id)])
    def get_warn_limit(self, tg_chat_id):
        return process_cmd('''SELECT warn_limit FROM chats WHERE tg_chat_id = %s''', values=[str(tg_chat_id)])[0][0]
    def delete_all_warns(self, username, tg_chat_id):
        chat_id = self.get_chat_id_by_tg_chat_id(tg_chat_id)
        process_cmd('''DELETE FROM warns WHERE user_id = %s AND chat_id = %s''', values = [self.get_id_by_name(username, tg_chat_id), chat_id])
        process_cmd('''UPDATE users SET warn_count = 0 WHERE username = %s AND chat_id = %s''', [username, chat_id])

database = Database()
database.new_chat('dfsf', '123')

database.new_chat('dfsf', '123')

