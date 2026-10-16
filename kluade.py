import json
import math
import os
import random


class CharRNN:

    def __init__(self, hidden_size=64, seq_length=25, lr=0.01):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lr = lr

        # Vocabulario de caracteres base (se expande dinámicamente)
        self.chars = list(
            " abcdefghijklmnopqrstuvwxyz0123456789áéíóúñ!?,.:;\n"
        )
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Inicialización de Pesos de la RNN
        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        # y_t = W_hy * h_t + b_y
        self.init_weights()

    def init_weights(self):
        scale = 0.1
        self.Wxh = [
            [random.uniform(-scale, scale) for _ in range(self.hidden_size)]
            for _ in range(self.vocab_size)
        ]
        self.Whh = [
            [random.uniform(-scale, scale) for _ in range(self.hidden_size)]
            for _ in range(self.hidden_size)
        ]
        self.Why = [
            [random.uniform(-scale, scale) for _ in range(self.vocab_size)]
            for _ in range(self.hidden_size)
        ]

        self.bh = [0.0] * self.hidden_size
        self.by = [0.0] * self.vocab_size

    def add_unknown_chars(self, text):
        """Añade caracteres nuevos al vocabulario si aparecen."""
        updated = False
        for ch in text.lower():
            if ch not in self.char_to_ix:
                self.chars.append(ch)
                idx = len(self.chars) - 1
                self.char_to_ix[ch] = idx
                self.ix_to_char[idx] = ch
                updated = True

        if updated:
            old_vocab = self.vocab_size
            self.vocab_size = len(self.chars)
            # Reajustar matrices de pesos para el nuevo tamaño de vocabulario
            for _ in range(self.vocab_size - old_vocab):
                self.Wxh.append(
                    [
                        random.uniform(-0.1, 0.1)
                        for _ in range(self.hidden_size)
                    ]
                )
                self.by.append(0.0)
            for j in range(self.hidden_size):
                for _ in range(self.vocab_size - old_vocab):
                    self.Why[j].append(random.uniform(-0.1, 0.1))

    def train_step(self, text):
        """Entrenamiento BPTT (Backpropagation Through Time) sobre una secuencia de texto."""
        self.add_unknown_chars(text)
        text = text.lower()

        if len(text) < 2:
            return

        inputs = [self.char_to_ix[ch] for ch in text[:-1]]
        targets = [self.char_to_ix[ch] for ch in text[1:]]

        # Pass Forward
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = [0.0] * self.hidden_size

        for t in range(len(inputs)):
            xs[t] = [0.0] * self.vocab_size
            xs[t][inputs[t]] = 1.0  # One-hot vector

            # Capa oculta recurrente
            hs[t] = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                sum_h = self.bh[j]
                for i in range(self.vocab_size):
                    sum_h += xs[t][i] * self.Wxh[i][j]
                for k in range(self.hidden_size):
                    sum_h += hs[t - 1][k] * self.Whh[k][j]
                hs[t][j] = math.tanh(sum_h)

            # Capa de salida (Softmax)
            ys[t] = [0.0] * self.vocab_size
            for k in range(self.vocab_size):
                sum_y = self.by[k]
                for j in range(self.hidden_size):
                    sum_y += hs[t][j] * self.Why[j][k]
                ys[t][k] = sum_y

            # Softmax
            max_y = max(ys[t])
            exp_y = [math.exp(v - max_y) for v in ys[t]]
            sum_exp = sum(exp_y)
            ps[t] = [e / sum_exp for e in exp_y]

        # Pass Backward (BPTT)
        dWxh = [
            [0.0] * self.hidden_size for _ in range(self.vocab_size)
        ]
        dWhh = [
            [0.0] * self.hidden_size for _ in range(self.hidden_size)
        ]
        dWhy = [
            [0.0] * self.vocab_size for _ in range(self.hidden_size)
        ]
        dbh = [0.0] * self.hidden_size
        dby = [0.0] * self.vocab_size
        dhnext = [0.0] * self.hidden_size

        for t in reversed(range(len(inputs))):
            dy = list(ps[t])
            dy[targets[t]] -= 1.0  # Gradiente de Cross-Entropy

            for j in range(self.hidden_size):
                for k in range(self.vocab_size):
                    dWhy[j][k] += hs[t][j] * dy[k]
            for k in range(self.vocab_size):
                dby[k] += dy[k]

            dh = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                dh[j] = dhnext[j]
                for k in range(self.vocab_size):
                    dh[j] += self.Why[j][k] * dy[k]

            # Gradiente a través de tanh
            dhraw = [(1.0 - hs[t][j] ** 2) * dh[j] for j in range(self.hidden_size)]

            for j in range(self.hidden_size):
                dbh[j] += dhraw[j]
                for i in range(self.vocab_size):
                    dWxh[i][j] += xs[t][i] * dhraw[j]
                for k in range(self.hidden_size):
                    dWhh[k][j] += hs[t - 1][k] * dhraw[j]

            dhnext = [0.0] * self.hidden_size
            for k in range(self.hidden_size):
                for j in range(self.hidden_size):
                    dhnext[k] += self.Whh[k][j] * dhraw[j]

        # Actualización de pesos con Gradient Clipping (evita exploción de gradientes)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            if isinstance(dparam[0], list):
                for row in dparam:
                    for i in range(len(row)):
                        row[i] = max(-5.0, min(5.0, row[i]))
            else:
                for i in range(len(dparam)):
                    dparam[i] = max(-5.0, min(5.0, dparam[i]))

        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Wxh[i][j] -= self.lr * dWxh[i][j]
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.Whh[i][j] -= self.lr * dWhh[i][j]
            for k in range(self.vocab_size):
                self.Why[i][k] -= self.lr * dWhy[i][k]
        for j in range(self.hidden_size):
            self.bh[j] -= self.lr * dbh[j]
        for k in range(self.vocab_size):
            self.by[k] -= self.lr * dby[k]

    def sample(self, seed_text, length=50, temperature=0.7):
        """Genera texto carácter por carácter usando el estado de la red."""
        self.add_unknown_chars(seed_text)
        seed_text = seed_text.lower()

        h = [0.0] * self.hidden_size
        x = [0.0] * self.vocab_size

        # Calentar el estado oculto con la entrada del usuario
        for ch in seed_text:
            idx = self.char_to_ix[ch]
            x = [0.0] * self.vocab_size
            x[idx] = 1.0

            h_next = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                sum_h = self.bh[j]
                for i in range(self.vocab_size):
                    sum_h += x[i] * self.Wxh[i][j]
                for k in range(self.hidden_size):
                    sum_h += h[k] * self.Whh[k][j]
                h_next[j] = math.tanh(sum_h)
            h = h_next

        # Generar nuevos caracteres
        generated = ""
        curr_idx = self.char_to_ix[seed_text[-1]] if seed_text else 0

        for _ in range(length):
            x = [0.0] * self.vocab_size
            x[curr_idx] = 1.0

            h_next = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                sum_h = self.bh[j]
                for i in range(self.vocab_size):
                    sum_h += x[i] * self.Wxh[i][j]
                for k in range(self.hidden_size):
                    sum_h += h[k] * self.Whh[k][j]
                h_next[j] = math.tanh(sum_h)
            h = h_next

            y = [0.0] * self.vocab_size
            for k in range(self.vocab_size):
                sum_y = self.by[k]
                for j in range(self.hidden_size):
                    sum_y += h[j] * self.Why[j][k]
                y[k] = sum_y / temperature

            max_y = max(y)
            exp_y = [math.exp(v - max_y) for v in y]
            sum_exp = sum(exp_y)
            probs = [e / sum_exp for e in exp_y]

            # Muestreo estocástico
            r = random.random()
            cum_prob = 0.0
            curr_idx = 0
            for i, p in enumerate(probs):
                cum_prob += p
                if r <= cum_prob:
                    curr_idx = i
                    break

            char = self.ix_to_char[curr_idx]
            if char == "\n":
                break
            generated += char

        return generated


class KluadeRNNBot:

    def __init__(self, brain_file="kluade_rnn_weights.json"):
        self.brain_file = brain_file
        self.rnn = CharRNN(hidden_size=64, lr=0.02)
        self.last_user_text = ""
        self.train_count = 0
        self.MAX_TRAIN_STEPS = 1000  # Límite de pasos para evitar sobrecarga

        self.load_brain()

        # Entrenamiento inicial básico en el modelo (cadena de caracteres pura)
        base_corpus = [
            "hola kluade",
            "hola soy una red neuronal",
            "como estas estoy bien",
            "quien eres soy kluade rnn",
        ]
        if self.train_count == 0:
            for text in base_corpus:
                for _ in range(10):
                    self.rnn.train_step(text)

    def interact(self, user_text):
        # 1. Aprender de la interacción previa (Secuencia: Usuario -> Bot)
        if self.last_user_text and self.train_count < self.MAX_TRAIN_STEPS:
            training_pair = f"{self.last_user_text} -> {user_text}"
            for _ in range(5):  # 5 iteraciones BPTT por mensaje
                self.rnn.train_step(training_pair)
                self.train_count += 1
            self.save_brain()

        self.last_user_text = user_text

        # 2. Generar respuesta inferida por la Red Neuronal Recurrente
        response = self.rnn.sample(seed_text=user_text + " -> ", length=40)
        return response.strip()

    def save_brain(self):
        data = {
            "Wxh": self.rnn.Wxh,
            "Whh": self.rnn.Whh,
            "Why": self.rnn.Why,
            "bh": self.rnn.bh,
            "by": self.rnn.by,
            "chars": self.rnn.chars,
            "train_count": self.train_count,
        }
        try:
            with open(self.brain_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_brain(self):
        if os.path.exists(self.brain_file):
            try:
                with open(self.brain_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.rnn.chars = data["chars"]
                    self.rnn.char_to_ix = {
                        ch: i for i, ch in enumerate(self.rnn.chars)
                    }
                    self.rnn.ix_to_char = {
                        i: ch for i, ch in enumerate(self.rnn.chars)
                    }
                    self.rnn.vocab_size = len(self.rnn.chars)

                    self.rnn.Wxh = data["Wxh"]
                    self.rnn.Whh = data["Whh"]
                    self.rnn.Why = data["Why"]
                    self.rnn.bh = data["bh"]
                    self.rnn.by = data["by"]
                    self.train_count = data.get("train_count", 0)
            except Exception:
                pass


def main():
    bot = KluadeRNNBot()
    print("=" * 50)
    print("  KLUADE - RED NEURONAL RECURRENTE (CHAR-RNN)")
    print("=" * 50)
    print("Modelo 100% matemático sin reglas de texto.")
    print("Al inicio responderá balbuceos hasta que entrene con tus textos.\n")

    while True:
        try:
            user_input = input("Tú: ")
            if not user_input.strip():
                continue

            if user_input.lower().strip() in ["salir", "exit"]:
                print("Guardando pesos neuronales... ¡Adiós!")
                break

            reply = bot.interact(user_input)
            print(f"Kluade (RNN): {reply}\n")

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
