from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client


async def main() -> None:
    # Находим рядом лежащий файл сервера.
    # Так клиент не зависит от того, из какой папки вы его запускаете.
    server_script = Path(__file__).with_name("day_16_server.py")

    # Здесь описываем, КАК именно клиент должен запустить MCP-сервер.
    # В нашем случае это обычная локальная команда:
    # python3 homeworks/src/day_16_server.py
    server_params = StdioServerParameters(
        command="python3",
        args=[str(server_script)],
    )

    print("1. Запускаем локальный MCP-сервер и открываем соединение...")

    # stdio_client:
    # - запускает сервер как подпроцесс,
    # - подключает клиента к stdin/stdout этого процесса,
    # - отдает потоки чтения/записи для MCP-сообщений.
    async with stdio_client(server_params) as (read_stream, write_stream):
        print("2. Соединение открыто. Создаем MCP-сессию...")

        # ClientSession - это объект, через который клиент общается с сервером.
        # Именно у него мы вызовем initialize() и list_tools().
        async with ClientSession(read_stream, write_stream) as session:
            print("3. Выполняем MCP initialize()...")

            # initialize() - это обязательный стартовый шаг MCP.
            # После него клиент и сервер "договариваются", что они готовы работать.
            init_result = await session.initialize()

            print("4. MCP-соединение установлено.")
            print(f"   Имя сервера: {init_result.serverInfo.name}")
            print(f"   Версия протокола: {init_result.protocolVersion}")

            print("5. Запрашиваем у сервера список доступных инструментов...")
            tools_result = await session.list_tools()

            print(f"6. Получено инструментов: {len(tools_result.tools)}")
            for index, tool in enumerate(tools_result.tools, start=1):
                print(f"   {index}. {tool.name} - {tool.description}")


if __name__ == "__main__":
    # anyio.run(...) запускает async main() из обычного Python-скрипта.
    anyio.run(main)
