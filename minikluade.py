import math
import random


class MiniRNN:

    def __init__(self, h_size=32):
        self.h_size = h_size
        self.chars = list(" abcdefghijklmnopqrstuvwxyzáéíóúñ!,.")
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for i, c in enumerate(self.chars)}
        v = len(self.chars)

        # Pesos y sesgos compactos
        self.Wxh = [
            [random.uniform(-0.2, 0.2) for _ in range(h_size)] for _ in range(v)
        ]
        self.Whh = [
            [random.uniform(-0.2, 0.2) for _ in range(h_size)] for _ in range(h_size)
        ]
        self.Why = [
            [random.uniform(-0.2, 0.2) for _ in range(v)] for _ in range(h_size)
        ]
        self.bh, self.by = [0.0] * h_size, [0.0] * v

    def expand(self, text):
        for ch in text.lower():
            if ch not in self.c2i:
                self.c2i[ch] = len(self.chars)
                self.i2c[len(self.chars)] = ch
                self.chars.append(ch)
                self.Wxh.append([random.uniform(-0.1, 0.1)] * self.h_size)
                for row in self.Why:
                    row.append(random.uniform(-0.1, 0.1))
                self.by.append(0.0)

    def step(self, text, lr=0.03):
        self.expand(text)
        text = text.lower()
        if len(text) < 2:
            return
        inp, tgt = [self.c2i[c] for c in text[:-1]], [self.c2i[c] for c in text[1:]]
        h = [0.0] * self.h_size

        for t in range(len(inp)):
            # Forward
            x = [0.0] * len(self.chars)
            x[inp[t]] = 1.0
            h_next = [
                math.tanh(
                    self.bh[j]
                    + sum(x[i] * self.Wxh[i][j] for i in range(len(self.chars)))
                    + sum(h[k] * self.Whh[k][j] for k in range(self.h_size))
                )
                for j in range(self.h_size)
            ]
            y = [
                sum(h_next[j] * self.Why[j][k] for j in range(self.h_size))
                + self.by[k]
                for k in range(len(self.chars))
            ]
            max_y = max(y)
            probs = [
                math.exp(v - max_y) / sum(math.exp(o - max_y) for o in y)
                for v in y
            ]

            # Backward aproximado (Gradiente sobre salida y estado)
            dy = list(probs)
            dy[tgt[t]] -= 1.0
            for j in range(self.h_size):
                for k in range(len(self.chars)):
                    self.Why[j][k] -= lr * h_next[j] * dy[k]
            h = h_next

    def generate(self, seed, length=25):
        self.expand(seed)
        h = [0.0] * self.h_size
        idx = self.c2i.get(seed[-1].lower(), 0)

        # Calentar estado
        for ch in seed.lower():
            if ch in self.c2i:
                idx = self.c2i[ch]
                x = [0.0] * len(self.chars)
                x[idx] = 1.0
                h = [
                    math.tanh(
                        self.bh[j]
                        + sum(x[i] * self.Wxh[i][j] for i in range(len(self.chars)))
                        + sum(h[k] * self.Whh[k][j] for k in range(self.h_size))
                    )
                    for j in range(self.h_size)
                ]

        res = ""
        for _ in range(length):
            x = [0.0] * len(self.chars)
            x[idx] = 1.0
            h = [
                math.tanh(
                    self.bh[j]
                    + sum(x[i] * self.Wxh[i][j] for i in range(len(self.chars)))
                    + sum(h[k] * self.Whh[k][j] for k in range(self.h_size))
                )
                for j in range(self.h_size)
            ]
            y = [
                sum(h[j] * self.Why[j][k] for j in range(self.h_size))
                + self.by[k]
                for k in range(len(self.chars))
            ]
            max_y = max(y)
            probs = [
                math.exp(v - max_y) / sum(math.exp(o - max_y) for o in y)
                for v in y
            ]

            # Selección probabilística
            r, acu = random.random(), 0.0
            for i, p in enumerate(probs):
                acu += p
                if r <= acu:
                    idx = i
                    break
            res += self.i2c.get(idx, "?")
        return res


# Bucle de ejecución rápido
print("--- KLUADE MINI RNN ---")
bot = MiniRNN(h_size=24)
last = ""

while True:
    try:
        user = input("Tú: ").strip()
        if user.lower() in ["salir", "exit"]:
            break
        if not user:
            continue

        if last:
            bot.step(f"{last} -> {user}")
        last = user

        reply = bot.generate(user + " -> ", length=30)
        print(f"Kluade: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        break
