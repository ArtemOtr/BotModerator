<b>Бот модератор чатов с автоматической модерацией спама и токсичного общения</b>
<b>Доступные на данный момент модели для классификации текстов: SVM, LogisticRegression, RNN, MNBC</b>
<b>В будущем возможно добавление более востребованных средств NLP для классификации текстов, а также будут использованы инструменты CV для модерации изображений</b>

<b>Руководство для работы с ботом:</b>

/start - Начать работу с ботом.

/help - Получить информацию о доступных командах.


<b>Команды модерации:</b>
/ban &lt;причина&gt; - Заблокировать пользователя (использовать в ответ на сообщение).

/unban @username - Разблокировать пользователя.

/mute &lt;время&gt; &lt;причина&gt; - Замутить пользователя (использовать в ответ на сообщение). Пример времени: 1d 1h 1m 1s.

/unmute @username - Размутить пользователя.


<b>Предупреждения:</b>
/warn &lt;причина&gt; - Выдать предупреждение (использовать в ответ на сообщение).

/unwarn @username &lt;warn_id&gt; - Снять предупреждение. ID необязателен, в случае неуказания ID снимется последнее предупреждение.
(узнать warn_id можно с помощью команды /get_info_about_user, описанной ниже)

/set_warn_limit &lt;число&gt; - Установить лимит предупреждений.


<b>Информация:</b>
/get_info_about_user @username - Получить информацию о пользователе.

<b>Автоматическая модерация:</b>
/set_classification_parametrs - Прописывается в беседу, в которой администратор, добавивший бота, хочет изменить параметры автоматической модерациии. После этого администратору в ЛС отправляется сообщение для настройки
в выбранной беседе
