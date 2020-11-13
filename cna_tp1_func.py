#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1

archivo = 'cna_tp1_func.py'

Contiene funciones auxiliares para el programa principal.

Las funciones d_dx y d_dy generan matrices que operan según 'x' e 'y' la
derivación primera y segunda en diferencias finitas sobre el vector con el
estado actual deldominio.

La función cb_Dir aplica condición de borde Dirichlet sobre la matriz
final de la discretización del problema.
"""

import sys
import numpy as np
import scipy.sparse as sp

#**** FUNCIONES DEL PROGRAMA ****#


#%%
def d_dx(orden=None, n_el_x=1, n_el_y=1, delta_x=1, upwinding='NO'):
    """
    Función que devuelve una matriz para la derivada espacial según 'x' de
    un vector, normalmente el vector con la variable de interés del
    problema.

    La derivada puede ser de 1er o 2do orden, pudiendo crear matrices para
    derivadas con estrategia 'upwinding' (por defecto es sin 'upwinding').
    
    El vector sobre el que se aplica se ordena según elementos de coord
    'x' primero, y elementos de coord 'y' después. Por ejemplo, el
    elemento u_1,3 corresponde al nodo nx=1 ny = 3. Por este orden, las
    matrices de derivación según 'x' ocupan la diagonal principal y
    adyacentes.

    Se aplica automáticamente una condición de Neumann = 0 en las
    fronteras. Esto implica igualar, en la primera derivada, a 0 la fila
    correspodiente a ese elemento (u_0 o u_nx). Para la segunda derivada,
    los nodos fantasmas u_-1 y u_n+1 son deducidos de la ecuación para la
    primera derivada.

    Ejemplo de matriz de derivación orden 1 en 'x' sin upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es 0):

     0  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    -1  X  1  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0 -1  X  1 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 1 en 'x' con upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es 1):

     0  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    -1  X  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0 -1  X  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 2 en 'x' sin upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es -2):

     X  2  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     1  X  1  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  1  X  1 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  0  2  X | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 2 en 'x' con upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es -2. En los
    extremos, x es igual a -1):

     x  1  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     1  X  1  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  1  X  1 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
     0  0  1  x | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    ...

    Los bordes de la malla que son perpendiculares a la coordenada de
    derivación son tratados automáticamente con una condición de Neumann
    u_x = 0. En el diseño del programa, en el caso de que se aplique sobre
    ese borde una condición de Dirichlet, el valor fijo igual a 0 de la
    frontera sobreescribe en la matriz final las filas correspodientes.

    Por lo anterior, las esquinas del dominio (elementos u_0,0; u_0,ny;
    u_nx,0; u_nx,ny) tienen todas condición de borde de Neumann u_x = 0
    a menos que se aplique Dirichlet.
    
    Para armar la matriz final se construye una matriz base de nx*nx.
    En el caso de nx=4 sin upwinding, la matriz base tiene la forma:
    
     X  1  0  0     (El coeficiente X=0 corresponde al elemento para
     1  X  1  0      el que se calcula la derivada)
     0  1  X  1
     0  0  1  X
    
    Esta matriz luego hace producto kronecker con una matriz
    identidad de ny*ny elementos, para obtener la matriz final de
    nx*ny. Posteriormente se aplica la condición de Neumann en las filas
    correspondientes.
    """

    if orden==None:
        print("Error en función 'd_dx'")
        print("Orden de derivada no especificado")
        sys.exit(1)
    elif orden not in (1,2):
        print("Error en función d_dx")
        print("Orden de derivada mal especificado: debe ser <1> o <2>")
        sys.exit(1)
    else:
        pass

    if upwinding not in ("SI", "NO"):
        print("Error en función 'd_dx'")
        print("Upwinding mal especificado, debe ser 'SI' o 'NO'")
        sys.exit(1)
    else:
        pass

    # Construcción de las matrices de derivación m,n=nx*ny
    n_el_total = n_el_x * n_el_y

    if orden==1:
        coef_sec_inf = -1 # Coeficiente de diagonal para términos u_-1
        if upwinding=='SI':
            coef_sec_sup = 0 # Coeficiente de diagonal para términos u_+1
            coef_ppal = 1 # Coeficiente de diagonal para términos u_0
        else: # upwingind=='NO'
            coef_sec_sup = 1
            coef_ppal = 0


    else: # orden==2
        coef_ppal = -2
        coef_sec_sup = 1
        coef_sec_inf = 1
            
    # Armado de la matriz 'base' de nx*nx
    matriz_base = sp.diags([\
        [coef_ppal]*n_el_x,\
        [coef_sec_sup]*n_el_x,\
        [coef_sec_inf]*n_el_x],\
        offsets=[0,1,-1], format='lil')

    # En la primera derivada la aplicación de condición de Neumann u_x = 0
    # en los bordes perpendiculares a 'x' ('x_ini' y 'x_fin') implica
    # igualar toda la fila correspondiente al elemento a 0
    if orden==1:
        matriz_base[0], matriz_base[-1] = [0 * sp.eye(m=1,n=n_el_x)]*2
    
    # Aplicación de sustitución de nodos fantasmas, en la 2da derivada, en
    # los elementos correspodientes a frontera
    else: # orden==2
        if upwinding=='SI':
            matriz_base[0,0] += 1
            matriz_base[-1,-1] += 1
        else: # upwinding=='NO':
            matriz_base[0,1] += 1
            matriz_base[-1,-2] += 1


    # Armado de matriz final usando producto kronecker
    matriz_final = sp.kron(sp.identity(n_el_y),matriz_base,format='lil')

    # División de la matriz por el paso de discretización en 'x'
    if orden==1:
        if upwinding=='SI':
            matriz_final = (1/delta_x) * matriz_final
        else: # upwinding=='SI':
            matriz_final = (1/(2*delta_x)) * matriz_final

    else: # orden==2:
        matriz_final = (1/delta_x**2) * matriz_final

    # Fin función 'd_dx'
    return matriz_final
#%%

#%%
def d_dy(orden=None, n_el_x=1, n_el_y=1, delta_y=1, upwinding='NO'):
    """
    Función que devuelve una matriz para la derivada espacial según 'y' de
    un vector, normalmente el vector con la variable de interés del
    problema.

    La derivada puede ser de 1er o 2do orden, pudiendo crear matrices para
    derivadas con estrategia 'upwinding' (por defecto es sin 'upwinding').
    
    El vector sobre el que se aplica se ordena según elementos de coord
    'x' primero, y elementos de coord 'y' después. Por ejemplo, el
    elemento u_1,3 corresponde al nodo nx=1 ny = 3. Por este orden, las
    matrices de derivación según 'y' ocupan la diagonal principal y las
    adyacentes se encuentran desfazadas en nx elementos.

    Se aplica automáticamente una condición de Neumann = 0 en las
    fronteras. Esto implica igualar, en la primera derivada, a 0 la fila
    correspodiente a ese elemento (u_0 o u_n). Para la segunda derivada,
    los nodos fantasmas u_-1 y u_n+1 son deducidos de la ecuación para la
    primera derivada.

    Ejemplo de matriz de derivación orden 1 en 'y' sin upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es 0):

    X  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  X  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  X  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  0  X | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    -------------------------------------------------
   -1  0  0  0 | X  0  0  0 | 1  0  0  0 | 0  0  0  0
    0 -1  0  0 | 0  X  0  0 | 0  1  0  0 | 0  0  0  0
    0  0 -1  0 | 0  0  X  0 | 0  0  1  0 | 0  0  0  0
    0  0  0 -1 | 0  0  0  X | 0  0  0  1 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 1 en 'y' con upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es 1):

    X  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  X  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  X  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  0  X | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    -------------------------------------------------
   -1  0  0  0 | X  0  0  0 | 0  0  0  0 | 0  0  0  0
    0 -1  0  0 | 0  X  0  0 | 0  0  0  0 | 0  0  0  0
    0  0 -1  0 | 0  0  X  0 | 0  0  0  0 | 0  0  0  0
    0  0  0 -1 | 0  0  0  X | 0  0  0  0 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 2 en 'y' sin upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es -2):

    X  0  0  0 | 2  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  X  0  0 | 0  2  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  X  0 | 0  0  2  0 | 0  0  0  0 | 0  0  0  0
    0  0  0  X | 0  0  0  2 | 0  0  0  0 | 0  0  0  0
    -------------------------------------------------
   -1  0  0  0 | X  0  0  0 | 1  0  0  0 | 0  0  0  0
    0 -1  0  0 | 0  X  0  0 | 0  1  0  0 | 0  0  0  0
    0  0 -1  0 | 0  0  X  0 | 0  0  1  0 | 0  0  0  0
    0  0  0 -1 | 0  0  0  X | 0  0  0  1 | 0  0  0  0
    ...

    Ejemplo de matriz de derivación orden 2 en 'y' con upwinding, con
    condición de Neumann = 0, nx=4 y ny=4 (la X representa el elemento
    sobre el que se calcula la derivada, su coeficiente es -2. En los
    extremos, x es igual a -1):

    x  0  0  0 | 1  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  x  0  0 | 0  1  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  x  0 | 0  0  1  0 | 0  0  0  0 | 0  0  0  0
    0  0  0  x | 0  0  0  1 | 0  0  0  0 | 0  0  0  0
    -------------------------------------------------
   -1  0  0  0 | X  0  0  0 | 1  0  0  0 | 0  0  0  0
    0 -1  0  0 | 0  X  0  0 | 0  1  0  0 | 0  0  0  0
    0  0 -1  0 | 0  0  X  0 | 0  0  1  0 | 0  0  0  0
    0  0  0 -1 | 0  0  0  X | 0  0  0  1 | 0  0  0  0
    ...
    
    Los bordes de la malla que son perpendiculares a la coordenada de
    derivación son tratados automáticamente con una condición de Neumann
    u_y = 0. En el diseño del programa, en el caso de que se aplique sobre
    ese borde una condición de Dirichlet, el valor fijo igual a 0 de la
    frontera sobreescribe en la matriz final las filas correspodientes.

    Por lo anterior, las esquinas del dominio (elementos u_0,0; u_0,ny;
    u_nx,0; u_nx,ny) tienen todas condición de borde de Neumann u_y = 0
    a menos que se aplique Dirichlet.
    
    Para armar la matriz final se construye una matriz base de nx*nx.
    En el caso de nx=4 sin upwinding, la matriz base tiene la forma:
    
     X  0  0  0     (El coeficiente X=0 corresponde al elemento para
     0  X  0  0      el que se calcula la derivada)
     0  0  X  0
     0  0  0  X

    Para la matriz final primero se hace producto kronecker entre una
    matriz identidad de ny*ny elementos y la matriz base, y luego se
    insertan las diagonales adyacentes desfazadas nx y -nx elementos.

    Un ejemplo de matriz final orden 1, u_y = 0, sin upwinding, nx=ny=4
    (Las X denotan el elemento calculado, coeficiente igual a 0):

    X  0  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  X  0  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  X  0 | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    0  0  0  X | 0  0  0  0 | 0  0  0  0 | 0  0  0  0
    -------------------------------------------------
   -1  0  0  0 | X  0  0  0 | 1  0  0  0 | 0  0  0  0
    0 -1  0  0 | 0  X  0  0 | 0  1  0  0 | 0  0  0  0
    0  0 -1  0 | 0  0  X  0 | 0  0  1  0 | 0  0  0  0
    0  0  0 -1 | 0  0  0  X | 0  0  0  1 | 0  0  0  0
    -------------------------------------------------
    x  0  0  0 |-1  0  0  0 | X  0  0  0 | 1  0  0  0
    0  x  0  0 | 0 -1  0  0 | 0  X  0  0 | 0  1  0  0
    0  0  x  0 | 0  0 -1  0 | 0  0  X  0 | 0  0  1  0
    0  0  0  x | 0  0  0 -1 | 0  0  0  X | 0  0  0  1
    -------------------------------------------------
    0  0  0  0 | X  0  0  0 | 0  0  0  0 | X  0  0  0
    0  0  0  0 | 0  X  0  0 | 0  0  0  0 | 0  X  0  0
    0  0  0  0 | 0  0  X  0 | 0  0  0  0 | 0  0  X  0
    0  0  0  0 | 0  0  0  X | 0  0  0  0 | 0  0  0  X
    

    """

    if orden==None:
        print("Error en función 'd_dy'")
        print("Orden de derivada no especificado")
        sys.exit(1)
    elif orden not in (1,2):
        print("Error en función d_dy")
        print("Orden de derivada mal especificado: debe ser <1> o <2>")
        sys.exit(1)
    else:
        pass

    if upwinding not in ("SI", "NO"):
        print("Error en función 'd_dy'")
        print("Upwinding mal especificado, debe ser 'SI' o 'NO'")
        sys.exit(1)
    else:
        pass

    # Construcción de las matrices de derivación m,n=nx*ny
    n_el_total = n_el_x * n_el_y

    if orden==1:
        coef_sec_inf = -1 # Coeficiente de diagonal para términos u_-1
        if upwinding=='NO':
            coef_sec_sup = 1 # Coeficiente de diagonal para términos u_+1
            coef_ppal = 0 # Coeficiente de diagonal para términos u_0
        else: # upwingind=='SI'
            coef_sec_sup = 0
            coef_ppal = 1

    else: # orden==2
        coef_ppal = -2
        coef_sec_sup = 1
        coef_sec_inf = 1
            
    # Armado de la matriz 'base' de nx*nx
    matriz_base = sp.diags([coef_ppal]*n_el_x, offsets=0, format='lil')

    # Primer paso en armado de matriz final. Producto kronecker
    matriz_final = sp.kron(sp.identity(n_el_y),matriz_base,format='lil')

    # Segundo paso en armado de matriz final. Inserción de diagonales
    matriz_final.setdiag(values=coef_sec_sup, k=n_el_x)
    matriz_final.setdiag(values=coef_sec_inf, k=-n_el_x)

    # Último paso en armado de matriz final. Aplicación de condición
    # u_y = 0 en derivada primera en los bordes perpendiculares a 'y'
    # ('y_ini' e 'y_fin'), o sustitución de los nodos fantasmas u_-1 y
    # u_n+1 en la derivada segunda
    if orden==1:
        matriz_final = matriz_final.tolil()
        matriz_final[:n_el_x] = 0 * sp.eye(m=1,n=n_el_total)
        matriz_final[n_el_x*(n_el_y-1):] = 0 * sp.eye(m=1,n=n_el_total)

    else: # orden==2:
        if upwinding=="NO":
            sust_y_ini = sp.diags([1]*n_el_x,\
                offsets=n_el_x,\
                shape=(n_el_x,n_el_total),\
                format='lil')
            sust_y_fin = sp.diags([1]*n_el_x,\
                offsets=n_el_x*(n_el_y-2),\
                shape=(n_el_x,n_el_total),\
                format='lil')

            matriz_final[0:n_el_x] += sust_y_ini
            matriz_final[n_el_x*(n_el_y-1):] += sust_y_fin

        else: # upwinding=="SI"
            sust_y_ini = sp.diags([1]*n_el_x,\
                offsets=0,\
                shape=(n_el_x,n_el_total),\
                format='lil')
            sust_y_fin = sp.diags([1]*n_el_x,\
                offsets=n_el_x*(n_el_y-1),\
                shape=(n_el_x,n_el_total),\
                format='lil')

            matriz_final[0:n_el_x] += sust_y_ini
            matriz_final[n_el_x*(n_el_y-1):] += sust_y_fin

    # División de la matriz por el paso de discretización en 'y'
    if orden==1:
        if upwinding=='SI':
            matriz_final = (1/delta_y) * matriz_final
        else: # upwinding=='SI':
            matriz_final = (1/2*delta_y) * matriz_final

    else: # orden==2:
        matriz_final = (1/delta_y**2) * matriz_final

    # Fin función 'd_dy'
    return matriz_final
#%%

##%%
def cb_Dir(matriz_aplicacion, borde=None, valor=0,\
    n_el_x=1, n_el_y=1):
    """
    Función que aplica sobre una matriz los coeficientes necesarios para
    la condición de Dirichlet. Asegura que los valores del vector inicial
    se mantengan a lo largo de la simulación.
    
    Normalmente se aplicaría sobre la matriz final, es decir, sobre 
    aquella construida con la suma de los términos advectivos, difusivos, 
    de decaimiento y derivada temporal.

    Requiere que se especifique el borde aplicación.

    En el caso de que dos bordes adyacentes (comparten esquina), tegan
    condiciones de borde aplicadas distintas (uno Dirichlet y otro
    Neumann), en la esquina prevalece siempre la condición de Dirichlet.
    
    Modifica de a un borde por vez.
    """

    if borde==None:
        print("Error en función 'cb_Dir'")
        print("Borde para aplicación de condición de borde " +\
              "no especificado")
        sys.exit(1)
    elif borde not in ('x_ini','x_fin','y_ini','y_fin'):
        print("Error en función 'cb_Dir'")
        print("Borde de aplicación de condición de borde mal " +\
              "especificado. Debe ser alguna de estas opciones:\n" +\
              "'x_ini'\n'x_fin'\n'y_ini'\n'y_fin'")
        sys.exit(1)
    else:
        pass

    # Conversión de la matriz a formato 'lil'
    # (scipy de manera automática convierte a 'csr' luego de aplicar sumas
    # y eso impide la modificación de bordes que aquí se efectúa.)
    matriz_aplicacion = matriz_aplicacion.tolil()

    # Aplicación de la condición de borde
    n_el_total = n_el_x * n_el_y

    if borde=='y_ini':
        cb = sp.eye(m=n_el_x, n=n_el_total, k=0)
        matriz_aplicacion[0:n_el_x] = cb

    elif borde=='y_fin':
        cb = sp.eye(m=n_el_x, n=n_el_total, k=n_el_x*(n_el_y-1))
        matriz_aplicacion[n_el_x*(n_el_y-1):] = cb

    elif borde=='x_ini':
        for i in range(0,n_el_total,n_el_x):
            cb = sp.eye(m=1, n=n_el_total, k=i)
            matriz_aplicacion[i] = cb

    elif borde=='x_fin':
        for i in range(n_el_x-1,n_el_total,n_el_x):
            cb = sp.eye(m=1, n=n_el_total, k=i)
            matriz_aplicacion[i] = cb

    # Fin función 'cb_Dir'
    return matriz_aplicacion
#%%

#%%
def cb_vector_Forz(vector_solucion, valor_forzante,\
    pos_x=0, pos_y=0, n_el_x=1, n_el_y=1):
    """
    Función que aplica en el vector solución inicial el valor de la 
    forzante en el nodo correspodiente.
    
    Requiere que se especifique el nodo de aplicación.
    """

    if vector_solucion.any()==None:
        print("Error en función 'cb_For'")
        print("Vector solución  no especificado")
        sys.exit(1)

    elif valor_forzante==None:
        print("Error en función 'cb_For'")
        print("Valor de la forzante no especificado")
        sys.exit(1)
    else:
        pass

    # Número de elementos de la matriz final usada en el cálculo
    n_el_total = n_el_x * n_el_y

    # Matriz identidad
    matriz_I = sp.eye(m=n_el_total, format='lil')

    # Igualación a 0 de la fila de la matriz correspondiente al nodo de la
    # forzante en la matriz identidad
    fila_forzante = pos_x + n_el_x*np.int(pos_y)

    matriz_I[fila_forzante] = 0 * sp.eye(m=1, n=n_el_total)

    # Construcción del vector con valor de forzante en 
    # nodo correspondiente
    vector_forzante = 0 * sp.eye(m=n_el_total, n=1, format='lil')
    vector_forzante[fila_forzante] = valor_forzante
    vector_final = matriz_I.dot(vector_solucion) + vector_forzante

    # Fin función 'cb_vector_Forz'
    return vector_final
#%%

#%%
def cb_matriz_Forz(matriz_aplicacion, pos_x=0, pos_y=0,\
    n_el_x=1, n_el_y=1):
    """
    Función que modifica las matrices A y B (o LHS y RHS) para asegurar
    que el nodo correspodiente a la forzante mantenga su valor.

    Para lograr esto, la fila que contiene el nodo de la forzante es
    cambiada por una fila con un único coeficiente igual a 1 en el nodo
    correspondiente.
    """

    if matriz_aplicacion==None:
        print("Error en función 'cb_For'")
        print("Matriz de aplicación no especificada")
        sys.exit(1)

    # Número de elementos de la matriz final usada en el cálculo
    n_el_total = n_el_x * n_el_y

    # Conversión de la matriz a formato 'lil'
    # (scipy de manera automática convierte a 'csr' luego de aplicar 
    # sumas y eso impide la modificación de bordes que aquí se 
    # efectúa.)
    matriz_aplicacion = matriz_aplicacion.tolil()

    # Seteo a 1 del nodo correspodiente en la matriz de aplicación. Setea
    # igual a 0 los nodos restantes en la fila
    fila_forzante = pos_x + n_el_x*np.int(pos_y)

    matriz_aplicacion[fila_forzante] = \
        sp.eye(m=1, n=n_el_total, k=fila_forzante)

    # Fin función 'cb_matriz_Forz'
    return matriz_aplicacion
#%%

#%%
def conc_Cont(desc_cont=None, delta_x=1, delta_y=1, prof=1, delta_t=1):
    """
    Función que ...
    """
    pass
#%%

#%%
def auto_dt(delta_x=1, delta_y=1, incremento=0.5, t_final=1,
            lim_estabilidad=0.25, dif_long=1, dif_trans=1,
            seleccion="NO"):
    """
    Función que selecciona un paso de tiempo automáticamente en función de
    delta_x y delta_y, a modo de cumplir con la condición de estabilidad
    requerida para el método explícito.

    El incremento del paso temporal es por default en 0.5 seg. Puede modificarse
    a elección. Se sugieren pasos 'redondos' para facilitar la lectura de la
    solución.
    
    Se efectúa una resta entre el paso temporal que cumple la condición de
    estabilidad r_x o r_y =1/4 y un vector de los posibles tiempos entre
    
    Variables:
    'dif_long' es el coeficiente de difusividad longitudinal (alineado a 'x')
    'dif_trans' es coeficiente de difusividad transversal (alineado a 'y')
    'lim_estabilidad' es el límite superior para asegurar la estabilidad del
    método. Por defecto en 1/4 para problema bidimensional. Se puede modificar
    para cambiar el ajuste del 'dt'.
    """

    # Comprobación de parámetros
    if delta_t==None:
        print("Error en función 'auto_dt'")
        print("Intervalo 'dt' no especificado")
        sys.exit(1)
    else:
        pass
    
    # Comienzo de función
    if seleccion=="NO":
        print("\nSelección manual de 'DT'")
        pass
    else: # seleccion==SI:
        # Conversión a segundos de t_final
        t_final_seg = t_final * 60

        # Pasos temporales para comparar
        delta_ts = np.arange(incremento,
                             t_final_seg+incremento,
                             incremento)

        # Delta_t de prueba para delta_x y delta_y
        prueba_delta_tx = delta_x**2 * lim_estabilidad / dif_long
        prueba_delta_ty = delta_y**2 * lim_estabilidad / dif_trans

        # Resta de los delta_tx y delta_ty de prueba del arreglo con
        # los posibles pasos temporales
        delta_ts_x = delta_ts - prueba_delta_tx
        delta_ts_y = delta_ts - prueba_delta_ty

        # Selección del delta
        delta_tx = delta_ts[np.where(delta_ts_x<=0)].max()
        delta_ty = delta_ts[np.where(delta_ts_y<=0)].max()

        delta_t = np.minimum(delta_tx, delta_ty)

    # Fin de función 'auto_dt'
    return delta_t

#%%
 

#**** PRUEBA DE FUNCIONES ****#

#nx=6
#ny=4
#UW = "SI"

#np.set_printoptions(linewidth=1000, threshold=10000)

#D1X = d_dx(orden=1, n_el_x=nx, n_el_y=ny, upwinding=UW)
#print(D1X.toarray())

#print()

#D2X = d_dx(orden=2, n_el_x=nx, n_el_y=ny, upwinding=UW)
#print(D2X.toarray())

#D1Y = d_dy(orden=1, n_el_x=nx, n_el_y=ny, upwinding=UW)
#print(D1Y.toarray())

#print()

#D2Y = d_dy(orden=2, n_el_x=nx, n_el_y=ny, upwinding=UW)
#print(D2Y.toarray())

#**** FIN PROGRAMA ****#
