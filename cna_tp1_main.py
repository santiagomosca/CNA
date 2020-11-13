#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1

archivo: 'cna_tp1_main.py'

Programa principal que arma el sistema de diferencias finitas
y resuelve el problema de advección-difusión de un contaminante
en un río de geometría rectangular.
"""

import cna_tp1_func as cna_func
import cna_tp1_in as cna_in

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import os
import sys

def main(archivo_input):
    #if (len(sys.argv) != 2):
    #    print("Error: número incorrecto de argumentos")
    #    print("Modo de uso: " + __file__ + " <archivo_input>")
    #    sys.exit(1)

    print("Ejecutando programa " + __file__)

    #**** DATOS DEL PROBLEMA ****#

    # Lectura de variables del archivo input
    vs = cna_in.datos_input(archivo_input)

    # Geometría [metros]
    x_ini = vs['X_INI']
    x_fin = vs['X_FIN']
    y_ini = vs['Y_INI']
    y_fin = vs['Y_FIN']

    # Parámetros de modelado
    vel = vs['VEL'] # [m/s]
    h = vs['H']     # [m]
    f = vs['F']     # [adim]
    e_l = vs['E_L'] # [adim]
    e_t = vs['E_T'] # [adim]

    D_l = e_l * h * vel * f # difusividad longitudinal [m^2/s]
    D_t = e_t * h * vel * f # difusividad transversal [m^2/s]

    # Parámetros de la descarga de contaminante
    desc_cont = vs['DESC_CONT'] # Descarga continua [kg/día]
    c_dec = vs['C_DEC'] # Cte de decaimiento del contaminante [1/día]

    # Tiempo total a simular [minutos]
    t_total = vs['T_TOTAL']

    #**** DISCRETIZACIÓN DEL PROBLEMA ****#

    dx = vs['DX']
    dy = vs['DY']
    
    # Selección de paso temporal
    auto_dt = vs['AUTO_DT']
    
    if auto_dt=="NO":
        dt = vs['DT']

    else: # auto_dt=="SI"
        dt = cna_func.auto_dt(delta_x=dx, delta_y=dy,
                              t_final=t_total,
                              dif_long=D_l,dif_trans=D_t)

    Lx = x_fin - x_ini
    Ly = y_fin - y_ini

    print("\nDatos de discretización")
    print("-------------------")
    print("Longitud en 'x' del dominio: {:.1f} m".format(Lx))
    print("Longitud en 'y' del dominio: {:.1f} m".format(Ly))
    print("")
    print("Intervalo 'dt' de {:.1f} s".format(dt))
    print("Discretización en 'x' de {:.1f} m".format(dx))
    print("Discretización en 'y' de {:.1f} m".format(dy))
    print("")

    ## Los nodos de cálculo se ubican en el las esquinas de las celdas
    #nx = np.int(Lx/dx + 1)
    #ny = np.int(Ly/dy + 1)
    
    # Los nodos de cálculo se ubican en el centro de las celdas
    nx = np.int(Lx/dx)
    ny = np.int(Ly/dy)
    nt = np.int(nx*ny)

    print("Cantidad de nodos totales en 'x': {:d}".format(nx))
    print("Cantidad de nodos totales en 'y': {:d}".format(ny))
    print("")

    # Volumen de celda utilizado para calcular concentración por m^3
    v_cel = dx*dy*h # [m^3]

    #Conversión de unidades para descarga y decaimiento de contaminante
    cu = 1 / (3600 * 24) # día / (3600 seg/hora * 24 hora/día)

    # Aporte de contaminante
    cu_desc_cont = desc_cont * cu # [kg/seg]

    # Decaimiento de contaminante
    cu_c_dec = c_dec * cu # [1/seg]

    # Datos para estabilidad
    rx = D_l * dt / (dx**2)
    ry = D_t * dt / (dy**2)

    print("\nDatos para evaluar estabilidad:")
    print("    rx = {:.6f}".format(rx))
    print("    ry = {:.6f}".format(ry))

    # CONSTRUCCIÓN DE MATRICES
    # ========================

    # Selección de método
    #theta = 1.0 --> Fuertemente implícito
    #theta = 0.5 --> Crank-Nicolson
    #theta = 0.0 --> Explícito centrado
    theta = vs['THETA']
    
    print("\nCorriendo con theta = {:.1f}".format(theta))

    # Selección de upwinding
    upw = vs['UPWINDING']
    
    print("\nCorriendo con upwinding: {}".format(upw))

    # Matriz del término advectivo
    D1x = vel * cna_func.d_dx(
        orden=1, n_el_x=nx, n_el_y=ny, delta_x=dx, upwinding=upw)

    # Matrices de los términos difusivos
    D2x = D_l * cna_func.d_dx(
        orden=2, n_el_x=nx, n_el_y=ny, delta_x=dx, upwinding=upw)
    D2y = D_t * cna_func.d_dy(
        orden=2, n_el_x=nx, n_el_y=ny, delta_y=dy, upwinding=upw)

    # Matriz identidad
    I = sp.eye(nt)

    # Suma de matrices
    M = dt * (D2x  + D2y - I*cu_c_dec - D1x)

    # Matrices para solución LHS y RHS
    A = I - theta * M
    B = I + (1-theta) * M

    # Condiciones de borde (aplicación de Dirichlet = 0)
    # Por defecto los bordes presentan u_x = 0 si son normales
    # a la coordenada de la matriz de derivación dd_1
    
    # Recorre diccionario de variables buscando las que sean 'Dirichlet'
    cont_borde = 0
    for key in vs.keys():
        if vs[key] == 'DIR':
            # Registro de cantidad de bordes con condición Dirichlet
            cont_borde += 1
            # Construcción de la identificación del borde
            borde_aplicacion = key.lower().replace('cb_','')

            B = cna_func.cb_Dir(\
                B, borde=borde_aplicacion, n_el_x=nx, n_el_y=ny)
            A = cna_func.cb_Dir(\
                A, borde=borde_aplicacion, n_el_x=nx, n_el_y=ny)
                
        else:
            pass
        
    print("\nBordes con condición Dirichlet: {}".format(cont_borde))

    # Vector solución. Concentración inicial en todos los puntos igual a 0
    u_ini = np.zeros((nt,1))

    # Forzante. Ubicado en el nodo adyacente a la esquina
    # superior izquierda, sobre borde 'y_ini'
    xforz = 1
    yforz = 0
    
    ## ANTERIOR:
    ## La descarga de contaminante, unidades kg/s, se multiplica por el
    ## intervalo 'dt' para que en cada paso de tiempo la forzante sea igual
    ## a la cantidad de contaminante que se vuelca en el total del intervalo

    # ACTUAL
    # La descarga de contaminante, unidades kg/s, se aplica como tal como
    # forzante, expresando una 'cantidad instantánea'. Con esto, es posible
    # comparar la cantidad de contaminante en cada paso temporal
    # independientemente del 'dt' utilizado
    vforz = cu_desc_cont


    print("\nFuente: {:.3e} kg/s de contaminante".format(cu_desc_cont))
    #print("Descarga por intervalo 'dt' de {:.1f} s: {:.3e} kg".\
          #format(dt, vforz))


    # Aplicación de la forzante al vector de condición inicial y a las
    # matrices de cálculo correspondiente
    u_rhs = cna_func.cb_vector_Forz(\
        vector_solucion=u_ini,\
        pos_x=xforz,\
        pos_y=yforz,\
        valor_forzante=vforz,\
        n_el_x=nx,\
        n_el_y=ny)

    B = cna_func.cb_matriz_Forz(\
        matriz_aplicacion=B,\
        pos_x=xforz,\
        pos_y=yforz,\
        n_el_x=nx,\
        n_el_y=ny)
    A = cna_func.cb_matriz_Forz(\
        matriz_aplicacion=A,\
        pos_x=xforz,\
        pos_y=yforz,\
        n_el_x=nx,\
        n_el_y=ny)


    #**** IMPRESIÓN DE MATRICES FINALES ****#    
    np.set_printoptions(linewidth=1000)
    #print("Matriz 'B'")
    #print(B.toarray())
    #print("\n\n\n")
    #print("Matriz 'A'")
    #print(A.toarray())


    #**** SOLUCIÓN ITERATIVA ****#
    print("\nEjecutando bucle de solución:")
    print("-----------------------------")

    # Conversión del tiempo final a segundos
    t_final = t_total*60

    # Cantidad de pasos
    n_pasos = np.arange(dt, t_final+dt,dt)

    # Bucle de solución
    for t in n_pasos:

        #print("Vector 'u_rhs', t = {} s".format(t))
        #print(u_rhs)

        # Cálculo del vector solución
        u_n1 = splinalg.spsolve(A.tocsc(), B.dot(u_rhs))
        
        # Actualización del vector viejo
        u_rhs = u_n1

        # PARA PENSAR: la cantidad de contaminante usada como
        # forzante es la descarga, unidades kg/s, multiplicada por
        # el intervalo 'dt'. El vector forzante tiene, entonces,
        # unidades kg. ¿Es esto consistente con la forma de las
        # ecuaciones matriciales? Si del lado izquierdo se despeja
        # el vector u_n+1 para que el lado derecho tenga la
        # expresión r=D*dt/dx^2, entonces pareciera que sí.
        # Por el momento, este es el enfoque adoptado.
        #
        # El problema es que así, para distintos intervalos dt
        # se tienen valores de contaminante diferentes. Entonces,
        # para uniformizar la solución, se divide esto por 'dt',
        # lo cual parece deshacer lo descrito en el párrafo
        # anterior. ¿CUÁL ES EL PROCEDIMIENTO CORRECTO?
        #
        # Finalmente, para obtener la concentración, se divide
        # el valor anterior por el volumen del nodo, calculado
        # como dx*dy*h
        
        ## ANTERIOR
        #sol_concentracion = u_n1/(v_cel*dt)

        # ACTUAL
        sol_concentracion = u_n1/v_cel

        # Impresión tiempo cada 1 minuto
        if (t/60)%1==0:
            print("\nTiempo: {:.1f} min".format(t/60))
            print("Valor máximo de concentración instantánea: "\
                  + "{:.3e} kg/m^3".format(sol_concentracion.max()))

        # Guardar vector solución a archivo según el intervalo
        # especificado
        dir_sol = "cna_tp1_sol_dx{}_dy{}_dt{}_theta{}".\
            format(dx,dy,dt,theta)

        t_sol = vs['T_SOL'] # [min]

        if (t/60)%t_sol==0:
            arch_sol = "cont_{:.1f}".format(t)
            ruta_sol = os.path.join(os.getcwd(),dir_sol,arch_sol)

            try:
                os.mkdir(dir_sol)
            except OSError as error:
                pass

            encabezado = "Solución para concentración de " +\
            "contaminante [kg/m^3]\n" +\
            "Theta = {:.1f}\n".format(theta) +\
            "t = {:.2f} min".format(t/60)

            np.savetxt(ruta_sol,sol_concentracion.reshape(ny,nx),\
                       fmt='%.6e',header=encabezado)
            
    #**** FIN MAIN ****#
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Error: número incorrecto de argumentos")
        print("Modo de uso: " + __file__ + " <archivo_input>")
        sys.exit(1)
    else:
        main(sys.argv[1])
    
#**** FIN PROGRAMA ****#
