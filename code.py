import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Donnees ###

nu = 0.1
T = 10.


def m_0(x):
    if x < 0 or x > 1:
        return 0
    return np.sin(2 * np.pi * x)


m_0 = np.vectorize(m_0)


def u_T(x):
    if x < 0 or x > 1:
        return 0
    return - np.sin(np.pi * x)


u_T = np.vectorize(u_T)


def F(x):
    return 3 * np.sin(x);


F = np.vectorize(F)


### Calcul de m(x,t) ###

def M_0(N):
    x = np.linspace(0, 1, N + 1)
    centres = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
    return m_0(centres)


def mat_derivee_seconde(N):
    retour = np.diag(-2 * np.ones(N))
    retour += np.diag(np.ones(N - 1), k=1)
    retour += np.diag(np.ones(N - 1), k=-1)
    return retour


def mat_derivee_amont(N):
    retour = np.diag(-1 * np.ones(N))
    retour += np.diag(np.ones(N - 1), k=1)
    return retour


def mat_derivee_aval(N):
    retour = np.diag(np.ones(N))
    retour += np.diag(-1 * np.ones(N - 1), k=-1)
    return retour


def M_suivant(N, P, A, M_actuel):
    delta_t = T / (P - 1)
    I = np.identity(N)
    mat = ((1. / delta_t) * I) - A
    multiplication = (1. / delta_t) * np.linalg.inv(mat)
    return np.dot(multiplication, np.transpose(M_actuel))


def A_mat(N, mode):
    delta_x = 1. / N
    if mode == "amont":
        return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (1. / delta_x) * mat_derivee_amont(N)
    else:
        return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (1. / delta_x) * mat_derivee_aval(N)


def M(N, P, mode):
    A = A_mat(N, mode)
    retour = []
    retour.append(M_0(N))
    for i in range(P - 1):
        suivant = M_suivant(N, P, A, retour[-1])
        retour.append(suivant)

    return retour


def affiche_M(N, P, mode):
    donnees = M(N, P, mode)
    X = np.linspace(0, 1, N + 1)
    X = [(X[i] + X[i + 1]) / 2 for i in range(len(X) - 1)]
    t = np.linspace(0, T, P)
    espace = []
    temps = []
    ordonnee = []
    for te in range(len(t)):
        ligne = donnees[te]
        for x in range(len(X)):
            espace.append(X[x])
            temps.append(t[te])
            ordonnee.append(ligne[x])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(espace), np.array(temps), np.array(ordonnee))
    plt.show()


### Calcul de u(x,t) ###

def U_T(N):
    x = np.linspace(0, 1, N + 1)
    centres = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
    return u_T(centres)


def B_mat(N, mode):
    delta_x = 1. / N
    if mode == "amont":
        return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (1. / delta_x) * mat_derivee_amont(N)
    else:
        return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (1. / delta_x) * mat_derivee_aval(N)


def U_suivant(N, P, B, U_actuel, Fm):
    delta_t = T / (P - 1)
    I = np.identity(N)
    mat = ((1. / delta_t) * I) + B
    right = Fm + ((1. / delta_t) * U_actuel)
    return np.dot(np.linalg.inv(mat), np.transpose(right))


def U(N, P, mode, M):
    M_valeurs = np.copy(M)
    M_valeurs = np.flipud(M_valeurs)
    F_m = F(np.array(M_valeurs))
    B = B_mat(N, mode)
    retour = []
    retour.append(U_T(N))
    for i in range(P - 1):
        suivant = U_suivant(N, P, B, retour[-1], F_m[i + 1])
        retour.append(suivant)
    retour = np.flipud(retour)
    return retour


def affiche_M_U(N, P, mode):
    M_valeurs = M(N, P, mode)
    U_valeurs = U(N, P, mode, M_valeurs)
    X = np.linspace(0, 1, N + 1)
    X = [(X[i] + X[i + 1]) / 2 for i in range(len(X) - 1)]
    t = np.linspace(0, T, P)
    espace = []
    temps = []
    ordonnee1 = []
    ordonnee2 = []
    for te in range(len(t)):
        ligne_M = M_valeurs[te]
        ligne_U = U_valeurs[te]
        for x in range(len(X)):
            espace.append(X[x])
            temps.append(t[te])
            ordonnee1.append(ligne_M[x])
            ordonnee2.append(ligne_U[x])
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(np.array(espace), np.array(temps), np.array(ordonnee1))
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(np.array(espace), np.array(temps), np.array(ordonnee2))
    plt.show()


### POD sur M ###

def M_barre(M, r):
    M_valeurs = np.copy(M)
    M_valeurs = np.transpose(M_valeurs)
    gauche, val_p, droite = np.linalg.svd(M_valeurs)
    return gauche[:, :r]


def A_tilde(A, M_bar):
    return np.dot(np.transpose(M_bar), np.dot(A, M_bar))


def C_suivant(r, P, A, C_actuel):
    delta_t = T / (P - 1)
    I = np.identity(r)
    mat = ((1. / delta_t) * I) - A
    multiplication = (1. / delta_t) * np.linalg.inv(mat)
    return np.dot(multiplication, np.transpose(C_actuel))


def M_reduit(M, r, mode):
    P, N = np.array(M).shape
    M_base = M_barre(M, r)

    A = A_mat(N, mode)
    A_tild = A_tilde(A, M_base)

    C = []
    C.append(np.dot(np.transpose(M_base), np.transpose(M[0])))
    for i in range(P - 1):
        suivant = C_suivant(r, P, A_tild, C[-1])
        C.append(suivant)

    retour = []
    for i in range(P):
        retour.append(np.dot(M_base, np.transpose(C[i])))

    return retour


def affiche_M_Mtilde(N, P, r, mode):
    M_valeurs = M(N, P, mode)
    Mtilde_valeurs = M_reduit(M_valeurs, r, mode)
    X = np.linspace(0, 1, N + 1)
    X = [(X[i] + X[i + 1]) / 2 for i in range(len(X) - 1)]
    t = np.linspace(0, T, P)
    espace = []
    temps = []
    ordonnee1 = []
    ordonnee2 = []
    for te in range(len(t)):
        ligne_M = M_valeurs[te]
        ligne_Mtilde = Mtilde_valeurs[te]
        for x in range(len(X)):
            espace.append(X[x])
            temps.append(t[te])
            ordonnee1.append(ligne_M[x])
            ordonnee2.append(ligne_Mtilde[x])
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(np.array(espace), np.array(temps), np.array(ordonnee1))
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(np.array(espace), np.array(temps), np.array(ordonnee2))
    plt.show()


def erreur_M(r, M, mode):
    Mtilde = M_reduit(M, r, mode)
    return np.linalg.norm(np.array(M) - np.array(Mtilde)) ** 2


def trace_erreur_M(r_min, r_max, N, P, mode):
    r = np.arange(r_min, r_max + 1)
    M_val = M(N, P, mode)
    erreurs = [erreur_M(i, M_val, mode) for i in r]
    plt.plot(r, erreurs)
    plt.show()


### POD sur U ###

def U_barre(U, r):
    U_valeurs = np.copy(U)
    U_valeurs = np.transpose(U_valeurs)
    gauche, val_p, droite = np.linalg.svd(U_valeurs)
    return gauche[:, :r]


def B_tilde(B, U_bar):
    return np.dot(np.transpose(U_bar), np.dot(B, U_bar))


def M_tilde(M, U_bar):
    retour = np.dot(np.transpose(U_bar), np.transpose(M))
    retour = np.transpose(retour)
    retour = np.flipud(retour)
    return retour


def D_suivant(r, P, Bt, D_actuel, Fm):
    delta_t = T / (P - 1)
    I = np.identity(r)
    mat = ((1. / delta_t) * I) + Bt
    right = Fm + ((1. / delta_t) * D_actuel)
    return np.dot(np.linalg.inv(mat), np.transpose(right))


def U_reduit(U, r, mode, M):
    P, N = np.array(U).shape
    U_base = U_barre(U, r)

    B = B_mat(N, mode)
    B_tild = B_tilde(B, U_base)

    Fm_tild = M_tilde(F(np.array(M)), U_base)

    D = []
    D.append(np.dot(np.transpose(U_base), np.transpose(U_T(N))))
    for i in range(P - 1):
        suivant = D_suivant(r, P, B_tild, D[-1], Fm_tild[i + 1])
        D.append(suivant)

    retour = []
    for i in range(P):
        retour.append(np.dot(U_base, np.transpose(D[i])))

    retour = np.flipud(retour)

    return retour


def affiche_U_Utilde(N, P, r, mode, M):
    U_valeurs = U(N, P, mode, M)
    Utilde_valeurs = U_reduit(U_valeurs, r, mode, M)
    X = np.linspace(0, 1, N + 1)
    X = [(X[i] + X[i + 1]) / 2 for i in range(len(X) - 1)]
    t = np.linspace(0, T, P)
    espace = []
    temps = []
    ordonnee1 = []
    ordonnee2 = []
    for te in range(len(t)):
        ligne_U = U_valeurs[te]
        ligne_Utilde = Utilde_valeurs[te]
        for x in range(len(X)):
            espace.append(X[x])
            temps.append(t[te])
            ordonnee1.append(ligne_U[x])
            ordonnee2.append(ligne_Utilde[x])
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(np.array(espace), np.array(temps), np.array(ordonnee1))
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(np.array(espace), np.array(temps), np.array(ordonnee2))
    plt.show()


### Tests ###

N = 60
P = 20
r = 2
mode = "aval"

#affiche_M_U(N,P,mode)


affiche_M_Mtilde(N,P,r,mode)

trace_erreur_M(1,10,N,P,mode)

# M_fin = M(N,P,mode)
# M_red = M_reduit(M_fin,r,mode)
# affiche_U_Utilde(N,P,r,mode,M_red)
