###########################################################################################################
#
# Methodes numeriques pour les problemes de grande dimension
# 
# Resolution du systeme suivant : 
# (1) -delta_t u - delta_x u - nu delta_xx u = F(m)
# (2) -delta_t m - delta_x m + nu delta_xx m = 0
# (3) u(x,T) = u_T(x)
# (4) m(x,0) = m_0(x)
# (5) u(0,t) = u(1,t) = m(0,t) = m(1,t) = 0 pour tout t
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

# Coefficient de "viscosite" de l'equation
nu = 0.1

# Duree totale de la simulation
T = 10.

# H := x -> alpha * x
alpha = 0.1

# Parametres de la gaussienne utilisee dans m_0
mu = 1. / 2
sigma = 1 / 20.

###########################################################################################################
# Donnees
###########################################################################################################

def m_0(x):
	"""
		Répartition initiale de la densite m au temps t = 0
	"""
	if x < 0 or x > 1:
		return 0
	return np.sin(2 * np.pi * x)
	#return (1 / (np.sqrt(2 * np.pi) * sigma) ) * np.exp(- ((x - mu)**2) / (2 * sigma * sigma) )
m_0 = np.vectorize(m_0)

def u_T(x):
	"""
		Répartition finale du cout u au temps t = T
	"""
	if x < 0 or x > 1:
		return 0
	return - np.sin(np.pi*x)
	#return 0
u_T = np.vectorize(u_T)

def F(x):
	"""
		Definition de la fonction F de l'equation (1)
	"""	
	return 3*np.sin(x);
F = np.vectorize(F)

###########################################################################################################
# Calcul de m(x,t) 
###########################################################################################################

def M_0(N):
	"""
		Discretisation avec un pas spatial N de la condition initiale m(x,0)
	"""	
	x = np.linspace(0,1,N+1)
	centres = [ (x[i] + x[i+1])/2 for i in range(len(x)-1) ]
	return m_0(centres)

def mat_derivee_seconde(N):
	"""
		Matrice traduisant l'approxiation par differences finies :
		delta_xx M_i = (M_{i+1} - 2 M_i + M_{i-1}) / 2*(delta_x)**2
	"""	
	retour = np.diag(-2 * np.ones(N))
	retour += np.diag(np.ones(N-1),k=1)
	retour += np.diag(np.ones(N-1),k=-1)
	return retour

def mat_derivee_amont(N):
	"""
		Matrice traduisant l'approxiation par differences finies :
		delta_x M_i = ( M_{i+1} - M_i ) / delta_x
	"""	
	retour = np.diag(-1 * np.ones(N))
	retour += np.diag(np.ones(N-1),k=1)
	return retour

def mat_derivee_aval(N):
	"""
		Matrice traduisant l'approxiation par differences finies :
		delta_x M_i = ( M_i - M_{i-1} ) / delta_x
	"""	
	retour = np.diag(np.ones(N))
	retour += np.diag(-1 * np.ones(N-1),k=-1)
	return retour

def M_suivant(N,P,A,M_actuel):
	"""
		Calcule le vecteur M au temps t+1 sachant :
			le nombre d'intervalles de discretisation spatiale N
			le nombre d'intervalles de discretisation temporelle P
			la matrice A de l'equation matricielle
			le vecteur M_actuel au temps t
		par le schema d'Euler implicite
	"""	
	delta_t = T / (P - 1)
	I = np.identity(N)
	mat = ( (1. / delta_t) * I ) - A
	multiplication = (1. / delta_t) * np.linalg.inv(mat)
	return np.dot(multiplication,np.transpose(M_actuel))

def A_mat(N,mode):
	"""
		Calcule la matrice A de taille NxN du probleme matriciel equivalent a l'equation (2)
		-delta_t M + AM = 0
		en utilisant la matrice de derivation "aval" ou "amont" selon la valeur de mode

	"""	
	delta_x = 1. / N
	if mode == "amont":
		return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_amont(N)
	else:
		return (nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_aval(N)

def M(N,P,mode):
	"""
		Calcule les valeurs de M par elements finis avec N elements spatiaux et P elements temporels
		en utilisant le mode de derivation "amont" ou "aval"
		Renvoie un tableau de P lignes et N colonnes, dont la rangee t contient les N valeurs de M(t)
	"""	
	A = A_mat(N,mode)
	retour = []
	retour.append(M_0(N))
	for i in range(P-1):
		suivant = M_suivant(N,P,A,retour[-1])
		retour.append(suivant)

	return retour

def affiche_M(N,P,mode):
	"""
		Trace le resultat de la fonction precedente
	"""	
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
	"""
		Discretisation avec un pas spatial N de la condition finale u(x,T)
	"""	
	x = np.linspace(0,1,N+1)
	centres = [ (x[i] + x[i+1])/2 for i in range(len(x)-1) ]
	return u_T(centres)

def B_mat(N,mode):
	"""
		Calcule la matrice B de taille NxN du probleme matriciel equivalent a l'equation (1)
		-delta_t U + BU = F(M)
		en utilisant la matrice de derivation "aval" ou "amont" selon la valeur de mode

	"""	
	delta_x = 1. / N
	if mode == "amont":
		return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_amont(N)
	else:
		return (-nu / 2. / delta_x / delta_x) * mat_derivee_seconde(N) + (alpha / delta_x) * mat_derivee_aval(N)

def U_suivant(N,P,B,U_actuel,Fm):
	"""
		Calcule le vecteur U au temps t sachant :
			le nombre d'intervalles de discretisation spatiale N
			le nombre d'intervalles de discretisation temporelle P
			la matrice B de l'equation matricielle
			le vecteur U_actuel au temps t+1 (evolution backward)
			Le vecteur F(M) au temps t+1 
		par le schema d'Euler implicite
	"""	
	delta_t = T / (P - 1)
	I = np.identity(N)
	mat = ( (1. / delta_t) * I ) + B
	right = Fm + ((1. / delta_t) * U_actuel)
	return np.dot(np.linalg.inv(mat),np.transpose(right))

def U(N,P,mode,M):
	"""
		Calcule les valeurs de U par elements finis avec N elements spatiaux et P elements temporels
		en utilisant le mode de derivation "amont" ou "aval" et la matrice NxP de valeurs de M
		Renvoie un tableau de P lignes et N colonnes, dont la rangee t contient les N valeurs de U(t)
	"""	
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
	"""
		Affiche cote a cote l'evolution de M et celle de U,
		calculees avec la discretisation spatiale N et la discretisation 
		temporelle P, ainsi qu'avec le mode "aval" ou "amont"
	"""	
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
# POD sur M
###########################################################################################################

def M_barre(M,r):
	"""
		Renvoie la base d'approximation au rang r de la matrice M obtenue par
		resolution par les elements finis
	"""	
	M_valeurs = np.copy(M)
	M_valeurs = np.transpose(M_valeurs)
	gauche,val_p,droite = np.linalg.svd(M_valeurs)
	return gauche[:,:r]

def A_tilde(A,M_bar):
	"""
		Renvoie la matrice A_tilde, remplacant la matrice A dans l'équation vérifiée par C
		et dépendant de la base de réduction M_bar
	"""	
	return np.dot( np.transpose(M_bar), np.dot(A,M_bar) )

def C_suivant(r,P,A,C_actuel):
	"""
		Calcule le vecteur C au temps t+1 sachant :
			le rang d'approximation de la base réduite r
			le nombre d'intervalles de discretisation temporelle P
			la matrice A de l'equation matricielle verifiee par C
			le vecteur C_actuel au temps t
		par le schema d'Euler implicite
	"""	
	delta_t = T / (P - 1)
	I = np.identity(r)
	mat = ( (1. / delta_t) * I ) - A
	multiplication = (1. / delta_t) * np.linalg.inv(mat)
	return np.dot(multiplication,np.transpose(C_actuel))

def M_reduit(M,r,mode):
	"""
		Calcule la nouvelle matrice de valeurs de M, de taille PxN,
		en se basant sur les valeurs de M calculées par elements finis
		et le rang de l'approximation
	"""
	P,N = np.array(M).shape
	M_base = M_barre(M,r)

	A = A_mat(N,mode)
	A_tild = A_tilde(A,M_base)

	C = []
	C.append( np.dot( np.transpose(M_base), np.transpose(M[0]) ) )
	for i in range(P-1):
		suivant = C_suivant(r,P,A_tild,C[-1])
		C.append(suivant)

	retour = []
	for i in range(P):
		retour.append( np.dot(M_base, np.transpose(C[i]) ) )

	return retour

def affiche_M_Mtilde(N,P,r,mode):
	"""
		Affiche les valeurs de M calculees par elements finis,
		avec les parametres N et P, et a cote les valeurs de M
		calculees avec une POD de rang r
	"""
	M_valeurs = M(N,P,mode)
	Mtilde_valeurs = M_reduit(M_valeurs, r, mode)
	X = np.linspace(0,1,N+1)
	X = [ (X[i] + X[i+1])/2 for i in range(len(X)-1) ]
	t = np.linspace(0,T,P)
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

def erreur_M(r,M,mode):
	"""
		renvoie l'erreur (norme de Frobenius) entre la matrice fine M
		et l'approximation par POD de rang r
	"""
	Mtilde = M_reduit(M,r,mode)
	return np.linalg.norm(np.array(M) - np.array(Mtilde))**2

def trace_erreur_M(r_min,r_max,N,P,mode):
	"""
		trace l'erreur precedente entre r_min et r_max
	"""
	r = np.arange(r_min,r_max+1)
	M_val = M(N,P,mode)
	erreurs = [erreur_M(i,M_val,mode) for i in r]
	plt.plot(r,erreurs)
	plt.show()

###########################################################################################################
# POD sur U
###########################################################################################################

def U_barre(U,r):
	U_valeurs = np.copy(U)
	U_valeurs = np.transpose(U_valeurs)
	gauche,val_p,droite = np.linalg.svd(U_valeurs)
	return gauche[:,:r]

def B_tilde(B,U_bar):
	return np.dot( np.transpose(U_bar), np.dot(B,U_bar) )

def M_tilde(M,U_bar):
	retour = np.dot( np.transpose(U_bar), np.transpose(M))
	retour = np.transpose(retour)
	retour = np.flipud(retour)
	return retour

def D_suivant(r,P,Bt,D_actuel,Fm):
	delta_t = T / (P - 1)
	I = np.identity(r)
	mat = ( (1. / delta_t) * I ) + Bt
	right = Fm + ((1. / delta_t) * D_actuel)
	return np.dot(np.linalg.inv(mat),np.transpose(right))

def U_reduit(U,r,mode,M):
	P,N = np.array(U).shape
	U_base = U_barre(U,r)

	B = B_mat(N,mode)
	B_tild = B_tilde(B,U_base)

	Fm_tild = M_tilde( F( np.array(M) ),U_base)

	D = []
	D.append( np.dot( np.transpose(U_base), np.transpose(U_T(N)) ) )
	for i in range(P-1):
		suivant = D_suivant(r,P,B_tild,D[-1],Fm_tild[i+1])
		D.append(suivant)

	retour = []
	for i in range(P):
		retour.append( np.dot(U_base, np.transpose(D[i]) ) )

	retour = np.flipud(retour)

	return retour

def affiche_U_Utilde(N,P,r,mode,M):
	U_valeurs = U(N,P,mode,M)
	Utilde_valeurs = U_reduit(U_valeurs, r, mode, M)
	X = np.linspace(0,1,N+1)
	X = [ (X[i] + X[i+1])/2 for i in range(len(X)-1) ]
	t = np.linspace(0,T,P)
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

def erreur_U(r,U,mode,M):
	Utilde = U_reduit(U,r,mode,M)
	return np.linalg.norm(np.array(U) - np.array(Utilde))**2

def trace_erreur_U(r_min,r_max,N,P,mode,M):
	r = np.arange(r_min,r_max+1)
	U_val = U(N,P,mode,M)
	erreurs = [erreur_U(i,U_val,mode,M) for i in r]
	plt.plot(r,erreurs)
	plt.show()


###########################################################################################################
# Tests
###########################################################################################################

N = 60
P = 20
r = 3
mode = "aval"

#affiche_M_U(N,P,mode)

#affiche_M_Mtilde(N,P,r,mode)

#trace_erreur_M(1,10,N,P,mode)

M_fin = M(N,P,mode)
M_red = M_reduit(M_fin,r,mode)
#affiche_U_Utilde(N,P,r,mode,M_fin)
affiche_U_Utilde(N,P,r,mode,M_red)
#trace_erreur_U(1,10,N,P,mode,M_fin)

