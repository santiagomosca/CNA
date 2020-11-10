#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1

archivo: 'cna_tp1_graf.py'

Programa auxiliar que grafica cortes de la solución a lo largo del eje
'x'. Se puede elegir la coordenada 'y' para el corte en el archivo input.
"""

import cna_tp1_in as cna_in

import numpy as np
import matplotlib.pyplot as plt
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
    h = vs['H']     # [m]

    # Parámetros de la descarga de contaminante
    desc_cont = vs['DESC_CONT'] # Descarga continua [kg/día]

    # Tiempo total a simular [minutos]
    t_total = vs['T_TOTAL']

    #**** DISCRETIZACIÓN DEL PROBLEMA ****#

    dx = vs['DX']
    dy = vs['DY']
    dt = vs['DT']

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

    xs = np.arange(x_ini+dx/2, x_fin,dx)

    print("Cantidad de nodos totales en 'x': {:d}".format(nx))
    print("Cantidad de nodos totales en 'y': {:d}".format(ny))
    print("")

    # Volumen de celda utilizado para calcular concentración por m^3
    v_cel = dx*dy*h # [m^3]

    #Conversión de unidades para descarga y decaimiento de contaminante
    cu = 1 / (3600 * 24) # día / (3600 seg/hora * 24 hora/día)

    # Aporte de contaminante
    cu_desc_cont = desc_cont * cu # [kg/seg]

    ## Datos para estabilidad
    #rx = D_l * dt / (dx**2)
    #ry = D_t * dt / (dy**2)

    #print("\nDatos para evaluar estabilidad:")
    #print("    rx = {:.6f}".format(rx))
    #print("    ry = {:.6f}".format(ry))

    # Selección de método
    #theta = 1.0 --> Fuertemente implícito
    #theta = 0.5 --> Crank-Nicolson
    #theta = 0.0 --> Explícito centrado
    theta = vs['THETA']

    if theta==0.0:
        metodo = "Explícito centrado"
    elif theta==0.5:
        metodo = "Crank-Nicolson"
    elif theta==1.0:
        metodo = "Fuertemente implícito"
    else:
        pass
    
    print("\nCorrida con theta = {:.1f}".format(theta))

    # Selección de upwinding
    upw = vs['UPWINDING']
    
    print("\nCorrida con upwinding: {}".format(upw))

    #**** IMPRESIÓN ITERATIVA A GRÁFICO ****#
    # Dimensión de gráfico: 1D o 2D
    dim_img = np.int(vs['DIM_IMG'])

    if dim_img!=1:
        print("Impresión en gráfico implementada sólo en 1D")
        sys.exit(1)
    else:
        pass

    # Nodo 'y' a lo largo del que se grafica
    y_img = np.int(vs['Y_IMG'])

    print("\nEjecutando bucle de salida a gráfico para y = {:.1f} m:".
          format(y_img*dy))
    
    # Intervalo en el que se imprime gráfico
    t_sol = vs['T_SOL'] # [min]


    # Conversión del tiempo final a segundos
    t_final = t_total*60

    # Cantidad de pasos
    n_pasos = np.arange(dt, t_final+dt,dt)

    # Bucle de búsqueda de valores máximos y mínimos en la solución
    for t in n_pasos:
        ##TODO
        ## Directorio con las soluciones guardadas
        #dir_sol = "cna_tp1_sol_dx{}_dy{}_dt{}_theta{}".\
            #format(dx,dy,dt,theta)
        ## Archivo con datos de solución
        #solucion = np.loadtxt(ruta_sol)
        pass

    # Bucle de solución
    for t in n_pasos:

        # Directorio con las soluciones guardadas
        dir_sol = "cna_tp1_sol_dx{}_dy{}_dt{}_theta{}".\
            format(dx,dy,dt,theta)

        # Directorio de imágenes
        dir_img = os.path.join(dir_sol,"imagenes_{}D_y{:.1f}m".format(dim_img,y_img*dy))

        # Impresión según intervalo de tiempo     
        if (t/60)%t_sol==0:
            arch_sol = "cont_{:.1f}".format(t)
            arch_img = "sol_{:03d}min_y{:03.1f}m.png".format(np.int(t/60),y_img*dy)
            ruta_sol = os.path.join(os.getcwd(),dir_sol,arch_sol)
            ruta_img = os.path.join(os.getcwd(),dir_img,arch_img)

            try:
                os.mkdir(dir_img)
            except OSError as error:
                pass

            # Datos de solución para el gráfico
            solucion = np.loadtxt(ruta_sol)

            # Texto para el gráfico
            titulo = "C (t = {:5.1f} min, y = 0 m)".format(t/60)
            nom_x = "L$_x$ [m]"
            nom_y = "C [kg/m$^3$]"
            c_inf = 0
            orden = np.abs(np.int(np.floor(np.log10(cu_desc_cont/v_cel))))
            c_sup = np.around(cu_desc_cont/v_cel,decimals=orden)
            texto = "{}".format(metodo) +\
                    "\n$\Delta$x: {:04.1f} m".format(dx) +\
                    "\n$\Delta$y: {:04.1f} m".format(dy) +\
                    "\n$\Delta$t: {:.1f} s".format(dt)
            x_texto = (x_ini+x_fin)*0.875
            y_texto = (c_inf+c_sup)*0.5
            etiqueta="Concentración"

            # Impresión del gráfico
            plt.figure()
            plt.title(titulo)
            plt.xlabel(nom_x)
            plt.ylabel(nom_y)
            plt.ylim(c_inf,c_sup)
            plt.text(x_texto, y_texto, texto, 
                     va="center", ha="center")
            plt.plot(xs, solucion[y_img], "r", label=etiqueta)
            plt.legend(loc="upper right", framealpha=1)
            plt.savefig(ruta_img)
            plt.clf()
            plt.close()

    #**** FIN MAIN ****#
    
if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Error: número incorrecto de argumentos")
        print("Modo de uso: " + __file__ + " <archivo_input>")
        sys.exit(1)
    else:
        main(sys.argv[1])
    
#**** FIN PROGRAMA ****#
