import httpx
import socket

print("=" * 60)
print("HTTPX DEEP DEBUG")
print("=" * 60)

# Тест 1: HTTP/1.1 explicitly
print("\nTest 1: Force HTTP/1.1")
try:
    transport = httpx.HTTPTransport(http1=True, http2=False)
    client = httpx.Client(transport=transport, timeout=5)
    r = client.get("http://127.0.0.1:11434/api/tags")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.text[:100]}")
    client.close()
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Тест 2: Без keep-alive
print("\nTest 2: Disable keep-alive")
try:
    transport = httpx.HTTPTransport(http1=True, http2=False, keepalive_expiry=0)
    client = httpx.Client(
        transport=transport,
        timeout=5,
        headers={"Connection": "close"}
    )
    r = client.get("http://127.0.0.1:11434/api/tags")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.text[:100]}")
    client.close()
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Тест 3: Посмотрим какие заголовки отправляет httpx
print("\nTest 3: Check what headers httpx sends")
try:
    # Создадим простой сервер-ловушку
    import threading

    captured_request = []

    def capture_server():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 19999))
        server.listen(1)
        server.settimeout(3)

        try:
            conn, addr = server.accept()
            data = conn.recv(4096)
            captured_request.append(data.decode())
            # Отправляем минимальный ответ
            conn.send(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            conn.close()
        except:
            pass
        finally:
            server.close()

    # Запускаем сервер в фоне
    t = threading.Thread(target=capture_server)
    t.start()

    import time
    time.sleep(0.1)

    # Делаем запрос httpx
    try:
        r = httpx.get("http://127.0.0.1:19999/test", timeout=2)
    except:
        pass

    t.join(timeout=2)

    if captured_request:
        print("  HTTPX sends these headers:")
        print("-" * 40)
        print(captured_request[0])
        print("-" * 40)
except Exception as e:
    print(f"  Error: {e}")

# Тест 4: Сравним с curl
print("\nTest 4: What curl sends")
try:
    import subprocess
    result = subprocess.run(
        ["curl", "-v", "-s", "http://127.0.0.1:11434/api/tags"],
        capture_output=True,
        text=True
    )
    # verbose output идёт в stderr
    print("  Curl verbose:")
    for line in result.stderr.split('\n'):
        if line.startswith('>'):
            print(f"  {line}")
except Exception as e:
    print(f"  Error: {e}")

# Тест 5: Эмулируем curl точно
print("\nTest 5: Emulate curl exactly")
try:
    client = httpx.Client(
        http1=True,
        http2=False,
        timeout=5,
        headers={
            "Host": "127.0.0.1:11434",
            "User-Agent": "curl/8.0",
            "Accept": "*/*",
        },
        follow_redirects=True
    )
    # Убираем дефолтные заголовки
    r = client.get("http://127.0.0.1:11434/api/tags")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.text[:100]}")
    client.close()
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Тест 6: Limits
print("\nTest 6: With connection limits")
try:
    limits = httpx.Limits(max_keepalive_connections=0, max_connections=1)
    transport = httpx.HTTPTransport(http1=True, http2=False)
    client = httpx.Client(
        transport=transport,
        limits=limits,
        timeout=5
    )
    r = client.get("http://127.0.0.1:11434/api/tags")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.text[:100]}")
    client.close()
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Тест 7: IPv4 explicitly через transport
print("\nTest 7: Force IPv4 socket")
try:
    import httpcore

    # Кастомный транспорт только IPv4
    transport = httpx.HTTPTransport(
        http1=True,
        http2=False,
        local_address="0.0.0.0"  # Force IPv4
    )
    client = httpx.Client(transport=transport, timeout=5)
    r = client.get("http://127.0.0.1:11434/api/tags")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.text[:100]}")
    client.close()
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")