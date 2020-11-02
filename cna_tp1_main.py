#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1
"""

import cna_tp1_func as cna_func
import cna_tp1_in as cna_in

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
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
    dt = vs['DT']

    Lx = x_fin - x_ini
    Ly = y_fin - y_ini

    nx = np.int(Lx/dx + 1)
    ny = np.int(Ly/dy + 1)

    #Conversión de unidades para descarga y decaimiento de contaminante
    cu = 1 / (3600 * 24) # día / (3600 seg/hora * 24 hora/día)

    # Aporte de contaminante
    cu_desc_cont = desc_cont * cu # [kg/seg]

    # Decaimiento de contaminante
    cu_c_dec = c_dec * cu # [1/seg]

    # Datos para estabilidad
    rx = D_l * dt / (dx**2)
    ry = D_t * dt / (dy**2)

    print("Datos para evaluar estabilidad")
    print("rx = {:.6f}".format(rx))
    print("ry = {:.6f}".format(ry))

    # CONSTRUCCIÓN DE MATRICES
    # ========================

    # Selección de método
    #theta = 1.0 --> Fuertemente implícito
    #theta = 0.5 --> Crank-Nicolson
    #theta = 0.0 --> Explícito centrado
    theta = vs['THETA']

    # Matriz del término advectivo
    if vs['UPWINDING'] == 'SI':
        D1x = vel / dx * cna_func.dd_1(
            coord='x', n_el_x=nx, n_el_y=ny, upwinding='SI')

    elif vs['UPWINDING'] == 'NO':
        D1x = vel / (2*dx) * cna_func.dd_1(
            coord='x', n_el_x=nx, n_el_y=ny, upwinding='NO')


    # Matrices de los términos difusivos
    D2x = D_l / dx**2 * cna_func.dd_2(coord='x', n_el_x=nx, n_el_y=ny)
    D2y = D_t / dy**2 * cna_func.dd_2(coord='y', n_el_x=nx, n_el_y=ny)

    # Matriz identidad
    I = sp.eye(nx*ny)

    # Suma de matrices
    M = dt * (D2x  + D2y - I*cu_c_dec - D1x)

    # Matriz de coeficientes
    A = I - theta * M

    # Matriz de términos independientes
    B = I + (1-theta) * M

    # Vector solución
    u = np.zeros((nx*ny,1))

    # Condiciones de borde (aplicación de Dirichlet)
    if vs['CB_X_INI'] == 'DIR':
        B = cna_func.cb_Dir(B, borde='x_ini', n_el_x=nx, n_el_y=ny)
        if theta != 0.0:
            A = cna_func.cb_Dir(A, borde='x_ini', n_el_x=nx, n_el_y=ny)
        else:
            pass

    elif vs['CB_X_FIN'] == 'DIR':
        B = func.cb_Dir(B, borde='x_fin', n_el_x=nx, n_el_y=ny)
        if theta != 0.0:
            A = func.cb_Dir(A, borde='x_fin', n_el_x=nx, n_el_y=ny)

    elif vs['CB_Y_INI'] == 'DIR':
        B = cna_func.cb_Dir(B, borde='y_ini', n_el_x=nx, n_el_y=ny)
        if theta != 0.0:
            A = cna_func.cb_Dir(A, borde='y_ini', n_el_x=nx, n_el_y=ny)

    elif vs['CB_Y_FIN'] == 'DIR':
        B = cna_func.cb_Dir(B, borde='y_fin', n_el_x=nx, n_el_y=ny)
        if theta != 0.0:
            A = cna_func.cb_Dir(A, borde='y_fin', n_el_x=nx, n_el_y=ny)


    # Forzante. Ubicado en el nodo adyacente a la esquina
    # superior izquierda, sobre borde 'y_ini'
    print("Fuente: {} kg/dt".format(cu_desc_cont * dt))
    pos_x = 1
    pos_y = 0
    pos_forzante = nx*np.int(ny*pos_y) + pos_x

    u[pos_forzante] = cu_desc_cont * dt

    fila_forzante = sp.eye(m=1, n=nx*ny, k=pos_forzante)
    B[pos_y] = fila_forzante


    #**** SOLUCIÓN ITERATIVA ****#
    # Conversión del tiempo final a segundos
    t_final = np.int(t_total*60/dt) + 1

    for i in range(1,t_final):
        t = dt * i

        # Armado del vector dato
        Bu = B.dot(u)

        # Cálculo del vector solución
        u_n1 = spsolve(A, Bu)

        # Actualización del vector viejo
        u = u_n1

        # Impresión tiempo
        if (t/60)%5==0:
            print('Tiempo: {:.1f} min'.format(t/60))

        # Guardar vector solución a archivo
        archivo = 'cont_{:.1f}'.format(t)
        directorio = 'cna_tp1_sol'
        ruta = os.path.join(directorio, archivo)

        try:
            os.mkdir(directorio)
        except OSError as error: 
           pass

        encabezado = "Solución para concentración de contaminante [kg]\n" +\
                     "Theta = {:.1f}\n".format(theta) +\
                     "t = {:.2f} min".format(t/60)

        np.savetxt(ruta,u_n1.reshape(ny,nx),fmt='%.6e', header=encabezado)
    
    #**** FIN MAIN ****#
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Error: número incorrecto de argumentos")
        print("Modo de uso: " + __file__ + " <archivo_input>")
        sys.exit(1)
    else:
        main(sys.argv[1])
    
#**** FIN PROGRAMA ****#
