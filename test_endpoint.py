from backend.chat_storage import load_chats, save_chats

data = load_chats()
print(data)

data["test_session"] = [{"role": "user", "content": "hello"}]
save_chats(data)

print(load_chats())