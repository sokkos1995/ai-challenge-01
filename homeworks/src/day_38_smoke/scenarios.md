# Day 38 — UI smoke scenarios (demo web UI)

Target: `http://127.0.0.1:8765` — `python3 -m homeworks.todoist.webapp`  
Credentials: `demo` / `demo123`

## S1 — Login

1. Open `/login`
2. Fill username/password
3. Submit
4. Expect redirect to `/tasks` and badge «Вы вошли как demo»

## S2 — Create task

1. Being logged in on `/tasks`
2. Enter title `Smoke task day_38`
3. Click «Создать»
4. Expect flash about create and a row with that title

## S3 — Verify task in list

1. On `/tasks`
2. Assert `data-testid=task-list` contains `Smoke task day_38`
3. Assert status `[todo]`

## S6 — Complete task (feature: Завершить)

1. Click «Завершить» on the created row
2. Expect flash `Завершено: …`
3. Assert status becomes `[done]`
4. Assert кнопка «Завершить» больше не показывается для этой задачи

## S4 — Delete task

1. Click «Удалить» on the created row
2. Expect flash about delete
3. Assert list is empty (`tasks-empty`) or title gone

## S5 — Logout

1. Click «Выйти»
2. Expect `/login`
3. Opening `/tasks` without session redirects to login
