#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1
"""

import sys
import numpy as np
import scipy.sparse as sp

#**** FUNCIONES DEL PROGRAMA ****#


#%%
def dd_1(coord=None, n_el_x=1, n_el_y=1, upwinding=False):
    """
    Función que devuelve una matriz para la primera derivada espacial de
    un vector, normalmente el vector con la variable de interés del
    problema.

    Requiere que se especifique la coordenada de derivación, pudiendo
    crear matrices para derivadar con estrategia 'upwinding' (por defecto
    es sin 'upwinding').
    
    El vector sobre el que se aplica se ordena según elementos de coord
    'x' primero, y elementos de coord 'y' después. Por ejemplo, el
    elemento u_1,3 corresponde al nodo nx=1 ny = 3. Por este orden, las
    matrices de derivación según 'x' ocupan la diagonal principal y
    adyacentes, mientras que las matrices de derivación según 'y' ocupan
    la diagonal principal y las adyacentes se desfazan de ella según la
    cantidad de nodos en 'x'.

    Los bordes de la malla que son perpendiculares a la coordenada de
    derivación son tratados automáticamente con una condición de Neumann
    igual a 0. Esto es así porque, en el caso de que se aplique sobre ese
    borde una condición de Dirichlet, el valor ya estará fijado y no
    debería ser afectado por el cálculo de una derivada sobreimpuesta.

    Por lo anterior, las esquinas del dominio (elementos u_0,0; u_0,ny;
    u_nx,0; u_nx,ny) tienen todas condición de borde de Neumann igual a 0.
    """

    if coord==None:
        print("Error: dirección de derivada no especificada.")
        sys.exit(1)
    elif coord not in ('x','y'):
        print("Error: la dirección de derivación debe ser 'x' o 'y'.")
        sys.exit(1)
    else:
        pass

    # Construcción de las diagonales para las matrices 'célula' de
    # nx filas.
    # Luego se hace producto Kronecker entre estas células y una
    # matriz identidad de ny filas.
    n_el_total = n_el_x * n_el_y
    mat_blanco = sp.identity(n_el_y)
    diags = np.empty((3, n_el_total))

    if upwinding==True:
        coef_ppal = 1     # Coeficiente de diagonal principal
        coef_sec_sup = 0  # Coeficiente de diagonal 'secundaria' superior
    else:
        coef_ppal = 0
        coef_sec_sup = 1
    coef_sec_inf = -1 # Coeficiente de diagonal 'secundaria' inferior

    diag_ppal = coef_ppal * np.ones(n_el_x)
    diag_sec_sup = coef_sec_sup * np.ones(n_el_x)
    diag_sec_inf = coef_sec_inf * np.ones(n_el_x)
    
    # Ajuste de diagonales según derivada en 'x' o 'y'
    if coord=='x':
        # Posición de la diagonal ppal, offset de diag sec superior
        # y offset de diag sec inferior
        diag_sec_sup[-1] = diag_sec_inf[-1] = 0
        pos_diags = np.array([0,1,-1])

    elif coord=='y':
        pos_diags = np.array([0,n_el_x,-n_el_x])

    # Construción de la matriz de diagonales
    diags[0] = np.tile(diag_ppal,n_el_y)
    diags[1] = np.tile(diag_sec_sup,n_el_y)
    diags[2] = np.tile(diag_sec_inf,n_el_y)

    #Matriz rala final
    matriz = sp.diags(diags, pos_diags, format='lil')

    # Aplicación de condición de Neumann en los bordes perpendiculares
    # a la coordenada de derivación
    fila_neumann = 0 * sp.eye(m=1, n=n_el_total)

    if coord=='x':
        # Bordes x=0 y x=xL
        matriz[0::n_el_x] = matriz[n_el_x-1::n_el_x] = fila_neumann
      # Esquinas u_0,ny y u_nx,ny
    elif coord=='y':
        # Bordes y=0 y y=yL
        matriz[0:n_el_x] = matriz[n_el_x*(n_el_y-1):] = fila_neumann

    return matriz
#%%

#%%
def dd_2(coord=None, n_el_x=1, n_el_y=1):
    """
    Función que devuelve una matriz para la segunda derivada espacial de
    un vector, normalmente el vector con la variable de interés del
    problema.
    
    El vector sobre el que se aplican se ordena según elementos de coord 'x'
    primero, y elementos de coord 'y' después. Por ejemplo, el elemento
    u_1,3 corresponde al nodo nx=1 ny = 3. Por este orden, las matrices de
    derivación según 'x' ocupan la diagonal principal y adyacentes, mientras
    que las matrices de derivación según 'y' ocupan la diagonal principal
    y las adyacentes se desfazan de ella según la cantidad de nodos en 'x'.

    Los bordes de la malla que son perpendiculares a la coordenada de
    derivación son tratados automáticamente con una condición de Neumann
    igual a 0. Por lo tanto, el valor de la derivada segunda también es
    igualado a 0. Esto es así porque, en el caso de que se aplique sobre
    ese borde una condición de Dirichlet, el valor ya estará fijado y no
    debería ser afectado por el cálculo de una derivada sobreimpuesta.

    Por lo anterior, las esquinas del dominio (elementos u_0,0; u_0,ny;
    u_nx,0; u_nx,ny) tienen todas condición de borde de Neumann igual a 0.
    """

    if coord==None:
        print("Error: dirección de derivada no especificada.")
        sys.exit(1)
    elif coord not in ('x','y'):
        print("Error: la dirección de derivación debe ser 'x' o 'y'.")
        sys.exit(1)
    else:
        pass

    # Construcción de las diagonales para las matrices 'célula' de
    # nx filas.
    # Luego se hace producto Kronecker entre estas células y una
    # matriz identidad de ny filas.
    n_el_total = n_el_x * n_el_y
    mat_blanco = sp.identity(n_el_y)
    diags = np.empty((3, n_el_total))

    coef_ppal = -2 # Coeficiente de diagonal principal
    coef_sec = 1   # Coeficiente de diagonal 'secundaria'

    diag_ppal = coef_ppal * np.ones(n_el_x)
    # Diagonales 'secundarias' superior e inferior
    diag_sec_sup = diag_sec_inf = coef_sec * np.ones(n_el_x)
    
    # Ajuste de diagonales según derivada en 'x' o 'y'
    if coord=='x':
        # Posición de la diagonal ppal, offset de diag sec superior
        # y offset de diag sec inferior
        diag_sec_sup[-1] = diag_sec_inf[-1] = 0
        pos_diags = np.array([0,1,-1])

    elif coord=='y':
        pos_diags = np.array([0,n_el_x,-n_el_x])

    # Construción de la matriz de diagonales
    diags[0] = np.tile(diag_ppal,n_el_y)
    diags[1] = np.tile(diag_sec_sup,n_el_y)
    diags[2] = np.tile(diag_sec_inf,n_el_y)

    #Matriz rala final
    matriz = sp.diags(diags, pos_diags, format='lil')

    # Aplicación de condición de Neumann en los bordes perpendiculares
    # a la coordenada de derivación
    fila_neumann = 0 * sp.eye(m=1, n=n_el_total)

    if coord=='x':
        # Bordes x=0 y x=xL
        matriz[0::n_el_x] = matriz[n_el_x-1::n_el_x] = fila_neumann
      # Esquinas u_0,ny y u_nx,ny
    elif coord=='y':
        # Bordes y=0 y y=yL
        matriz[0:n_el_x] = matriz[n_el_x*(n_el_y-1):] = fila_neumann

    return matriz
#%%

#%%
def cb_Dir(matriz_aplicacion, borde=None, valor=0,\
               n_el_x=1, n_el_y=1):
    """
    Función que aplica sobre una matriz los coeficientes necesarios para
    la condición de Dirichlet con valor igual a 0 (se asume para este
    problema una concentración nula cuando se aplique esta condición).
    
    Normalmente se aplicaría sobre la matriz final, es decir, sobre aquella
    construida con la suma de los términos advectivos, difusivos, de
    decaimiento y derivada temporal.

    Requiere que se especifique el borde aplicación.
    
    Modifica de a un borde por vez.
    """

    if borde==None:
        print("Error: borde de aplicación de condición de borde no " +\
              "especificado.")
        sys.exit(1)
    elif borde not in ('x0','xL','y0','yL'):
        print("Error: borde de aplicación de condición de borde mal " +\
              "especificado. Debe ser alguna de estas opciones:\n" +\
              "'x0'\n'xL'\n'y0'\n'yL'")
        sys.exit(1)
    else:
        pass

    # Conversión de la matriz a formato 'lil'
    # (scipy de manera automática convierte a 'csr' luego de aplicar sumas
    # y eso impide la modificación de bordes que aquí se efectúa.)
    matriz_aplicacion = matriz_aplicacion.tolil()

    # Aplicación de la condición de borde
    n_el_total = n_el_x * n_el_y
    cb = 0 * sp.eye(m=1, n=n_el_total)

    if borde=='y0':
        matriz_aplicacion[0:n_el_x] = cb

    elif borde=='yL':
        matriz_aplicacion[n_el_x*(n_el_y-1):] = cb

    elif borde=='x0':
        matriz_aplicacion[0::n_el_x] = cb

    elif borde=='xL':
        matriz_aplicacion[n_el_x-1::n_el_x] = cb

    return matriz_aplicacion
#%%
#nx = 4
#ny = 4

#B = sp.eye(nx, format='lil')
#B = cb_Dir(B, borde='x0', n_el_x=nx, n_el_y=ny)
#print(B.toarray())

#**** FIN PROGRAMA ****#
