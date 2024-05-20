import numpy as np
import matplotlib.pyplot as plt

def sprawdz_wygrana(gracz1, gracz2):
    mapa_wygranych = {'P': 'N', 'N': 'K', 'K': 'P'}

    if gracz1 == gracz2:
        return 0  # Remis

    if mapa_wygranych[gracz1] == gracz2:
        return -1
    else:
        return 1

def wybierz_ruch(predykcja_przeciwnika):
    mapa_wygranych = {'P': 'N', 'N': 'P', 'K': 'P'}
    return mapa_wygranych[predykcja_przeciwnika]

t1 = ['P', 'K', 'N']
wyst = np.array([[2, 4, 1],
                 [0, 0, 4],
                 [4, 1, 2]])
prob_op = np.array([3, 4, 3])

kasa = []
stankasy = 0
n = 10000
state = 'P'

for i in range(n):
    pred = np.random.choice(t1, p=wyst[t1.index(state)] / sum(wyst[t1.index(state)]))
    print(pred)
    op_akc = np.random.choice(t1, p=prob_op/sum(prob_op))
    print(op_akc)
    print(wybierz_ruch(pred))
    stankasy += sprawdz_wygrana(wybierz_ruch(pred), op_akc)
    kasa.append(stankasy)
    wyst[t1.index(state)][t1.index(op_akc)] += 1
    state = op_akc

plt.plot(kasa)
plt.show()
print(len(kasa))
