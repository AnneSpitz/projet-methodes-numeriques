###########################################################################################################
#
# Methodes numeriques pour les problemes de grande dimension
# 
# Projet 
# 
# Raphael GRAFF-MENTZINGER, Mohammed Amine KHELDOUNI, Clement RIU, Anne SPITZ, Laurent THANWERDAS
# 
###########################################################################################################

import math
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

###########################################################################################################
# Donnees
###########################################################################################################

nu = 0.1

# Duree totale de la simulation
T = 0.1

# H := x -> alpha * x
alpha = 10.

# Parametres de la gaussienne
mu = 1. / 5
sigma = 1 / 20.

###########################################################################################################
# Donnees
###########################################################################################################

def m_0(x):
	if x < 0 or x > 1:
		return 0.
	if x>=0 and x<0.8:
		return np.exp(- ((x - 0.8)**2) / (2 * sigma * sigma) )
	if x >= 0.8 and x <= 0.9:
		return 1.
	if x>=0.9 and x<1:
		return np.exp(- ((x - 0.9)**2) / (2 * sigma * sigma) )
m_0 = np.vectorize(m_0)

def u_T(x):
	if x < 0 or x > 1:
		return 0
	#return - np.sin(np.pi*x)
	return 0
u_T = np.vectorize(u_T)

def F(x):
	return 3*np.sin(x);
F = np.vectorize(F)

###########################################################################################################
# Calcul de m(x,t) 
###########################################################################################################

def M_0(N):
	x = np.linspace(0,1,N+1)
	centres = [ (x[i] + x[i+1])/2 for i in range(len(x)-1) ]
	return m_0(centres)

def mat_derivee_seconde(N):
	retour = np.diag(-2 * np.ones(N))
	retour += np.diag(np.ones(N-1),k=1)
	retour += np.diag(np.ones(N-1),k=-1)
	return retour

def mat_derivee_amont(N):
	retour = np.diag(-1 * np.ones(N))
	retour += np.diag(np.ones(N-1),k=1)
	return retour

def mat_derivee_aval(N):
	retour = np.diag(np.ones(N))
	retour += np.diag(-1 * np.ones(N-1),k=-1)
	return retour

def M_suivant(N,P,A,M_actuel):
	delta_t = T / (P - 1)
	I = np.identity(N)
	mat = ( (1. / delta_t) * I ) - A
	multiplication = (1. / delta_t) * np.linalg.inv(mat)
	return np.dot(multiplication,np.transpose(M_actuel))

def A_mat(N,mode):
	delta_x = 1. / N
	if mode == "amont":
		return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_amont(N)
	else:
		return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_aval(N)

def M(N,P,mode):
	A = A_mat(N,mode)
	retour = []
	retour.append(M_0(N))
	for i in range(P-1):
		suivant = M_suivant(N,P,A,retour[-1])
		retour.append(suivant)

	return retour

def affiche_M(N,P,mode):
	donnees = M(N,P,mode)
	X = np.linspace(0,1,N+1)
	X = [ (X[i] + X[i+1])/2 for i in range(len(X)-1) ]
	t = np.linspace(0,T,P)
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

###########################################################################################################
# Calcul de u(x,t) 
###########################################################################################################

def U_T(N):
	x = np.linspace(0,1,N+1)
	centres = [ (x[i] + x[i+1])/2 for i in range(len(x)-1) ]
	return u_T(centres)

def B_mat(N,mode):
	delta_x = 1. / N
	if mode == "amont":
		return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_amont(N)
	else:
		return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_aval(N)

def U_suivant(N,P,B,U_actuel,Fm):
	delta_t = T / (P - 1)
	I = np.identity(N)
	mat = ( (1. / delta_t) * I ) + B
	right = Fm + ((1. / delta_t) * U_actuel)
	return np.dot(np.linalg.inv(mat),np.transpose(right))

def U(N,P,mode,M):
	M_valeurs = np.copy(M)
	M_valeurs = np.flipud(M_valeurs)
	F_m = F( np.array(M_valeurs) )
	B = B_mat(N,mode)
	retour = []
	retour.append(U_T(N))
	for i in range(P-1):
		suivant = U_suivant(N,P,B,retour[-1],F_m[i+1])
		retour.append(suivant)
	retour = np.flipud(retour)
	return retour

def affiche_M_U(N,P,mode):
	M_valeurs = M(N,P,mode)
	U_valeurs = U(N,P,mode,M_valeurs)
	X = np.linspace(0,1,N+1)
	X = [ (X[i] + X[i+1])/2 for i in range(len(X)-1) ]
	t = np.linspace(0,T,P)
	espace = []
	temps = []
	ordonnee1 = []
	ordonnee2 = []
	for te in range(len(t)):
		ligne_M = M_valeurs[te]
		ligne_U = U_valeurs[te]
		for x in range(len(X)):
			if X[x] > 0.09:
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

###########################################################################################################
# Affichage
###########################################################################################################

N = 120
P = 20
r = 4
mode = "aval"

affiche_M_U(N,P,mode)

