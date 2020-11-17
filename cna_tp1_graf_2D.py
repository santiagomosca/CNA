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
import cna_tp1_func as cna_func

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
    vel = vs['VEL'] # [m/s]
    h = vs['H']     # [m]
    f = vs['F']     # [adim]
    e_l = vs['E_L'] # [adim]
    e_t = vs['E_T'] # [adim]

    D_l = e_l * h * vel * f # difusividad longitudinal [m^2/s]
    D_t = e_t * h * vel * f # difusividad transversal [m^2/s]
    print(type(D_l))
    print(type(D_t))

    # Parámetros de la descarga de contaminante
    desc_cont = vs['DESC_CONT'] # Descarga continua [kg/día]

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

    xs = np.arange(x_ini+dx/2, x_fin,dx)
    ys = np.arange(y_ini+dy/2, y_fin,dy)

    print("Cantidad de nodos totales en 'x': {:d}".format(nx))
    print("Cantidad de nodos totales en 'y': {:d}".format(ny))
    print("")

    # Volumen de celda utilizado para calcular concentración por m^3
    v_cel = dx*dy*h # [m^3]

    #Conversión de unidades para descarga y decaimiento de contaminante
    cu = 1 / (3600 * 24) # día / (3600 seg/hora * 24 hora/día)

    # Aporte de contaminante
    cu_desc_cont = desc_cont * cu # [kg/seg]

    # Forzante
    vforz = cu_desc_cont * dt

    ## Datos para estabilidad
    #rx = D_l * dt / (dx**2)
    #ry = D_t * dt / (dy**2)

    # Selección de método
    #theta = 1.0 --> Fuertemente implícito
    #theta = 0.5 --> Crank-Nicolson
    #theta = 0.0 --> Explícito centrado
    theta = vs['THETA']

    if theta==0.0:
        metodo = "Explícito Centrado"
    elif theta==0.5:
        metodo = "Crank-Nicolson"
    elif theta==1.0:
        metodo = "Fuertemente Implícito"
    else:
        pass
    
    print("\nCorrida con theta = {:.1f}".format(theta))

    # Selección de upwinding
    upw = vs['UPWINDING']
    
    print("\nCorrida con upwinding: {}".format(upw))

    #**** IMPRESIÓN ITERATIVA A GRÁFICO ****#
    # Dimensión de gráfico: 1D o 2D
    dim_img = np.int(vs['DIM_IMG'])

    if dim_img==1:
        # Nodo 'y' a lo largo del que se grafica
        y_img = np.int(vs['Y_IMG'])

        print("\nEjecutando bucle de salida a gráfico 1D, y = {:.1f} m:".
              format(y_img*dy))

    else: #dim_img==2
        print("\nEjecutando bucle de salida a gráfico 2D")
    
    # Intervalo en el que se imprime gráfico
    t_sol = vs['T_SOL'] # [min]


    # Conversión del tiempo final a segundos
    t_final = t_total*60

    # Cantidad de pasos
    n_pasos = np.arange(dt, t_final+dt,dt)

    # Bucle de solución
    for t in n_pasos:

        # Directorio con las soluciones guardadas
        dir_sol = "cna_tp1_sol_dx{}_dy{}_dt{}_theta{}".\
            format(dx,dy,dt,theta)

        # Impresión según intervalo de tiempo     
        if (t/60)%t_sol==0:
            arch_sol = "sol_{:.1f}".format(t)

            # Directorio de imágenes y archivo imagen
            if dim_img == 1:
                dir_img = os.path.join(dir_sol,"imagenes_{}D_y{:.1f}m".format(dim_img,y_img*dy))
                arch_img = "sol_{:03d}min_y{:03.1f}m.png".format(np.int(t/60),y_img*dy)

            else:
                dir_img = os.path.join(dir_sol,"imagenes_{}D".format(dim_img))
                arch_img = "sol_{:03d}min_map.png".format(np.int(t/60))

            ruta_sol = os.path.join(os.getcwd(),dir_sol,arch_sol)
            ruta_img = os.path.join(os.getcwd(),dir_img,arch_img)

            try:
                os.mkdir(dir_img)
            except OSError as error:
                pass

            # Datos de solución para el gráfico
            solucion = np.loadtxt(ruta_sol)

            # ADIMENSIONAL
            # ------------
            conc_ini = vforz/v_cel # Concentración inicial de la forzante

            # Dimensiones de los gráficos
            c_inf = 0 # Nivel inferior de concentración relativa
            c_sup = 3 # Nivel superior de concentración relativa

            xs_adim = np.linspace(0,1,nx)
            
            if dim_img==1:
                plt.ylim(c_inf,c_sup)

            else: #dim_img==2
                y_sup = 0.25 # Un cuarto del dominio en 'y'
                lim_plot_y = np.int(ny*y_sup)
                ys_adim = np.linspace(0,y_sup,lim_plot_y)

            # Texto para gráficos
            texto = "C$_{forz}$: " + "{:.3e} kg/m$^3$".\
                    format(conc_ini) +\
                    "\n$\Delta$x: {:04.1f} m".format(dx) +\
                    "\n$\Delta$y: {:04.1f} m".format(dy) +\
                    "\n$\Delta$t: {:.1f} s".format(dt)

            x_texto = (xs_adim[-1] - xs_adim[0])*0.875
            etiqueta="C/C$_{forz}$"
            nom_x = "L/L$_x$"
            
            if dim_img==1:
                y_texto = (c_sup-c_inf)*0.5
                titulo = "Método {}\n".format(metodo) + \
                         "t = {:5.1f} min, y = {:03.1f}0 m".\
                         format(t/60, y_img*dy)
                nom_y = "C/C$_{forz}$"

            else: #dim_img==2
                y_texto = (ys_adim[-1] - ys_adim[0])*0.75
                titulo = "Método {}\n".format(metodo) + \
                         "t = {:5.1f} min".format(t/60)
                nom_y = "L/L$_y$"
            
            # Impresión del gráfico adimensional
            plt.figure(figsize=(12,6))
            plt.tight_layout()
            plt.title(titulo)
            plt.xlabel(nom_x)
            plt.ylabel(nom_y)

            if dim_img==1:
                plt.ylim(c_inf,c_sup)
                plt.text(x_texto, y_texto, texto,
                         va="center", ha="center")
                plt.plot(xs_adim,
                         solucion[y_img]/(conc_ini),
                         "r", label=etiqueta)
                plt.legend(loc="upper right", framealpha=1)

            else: #dim_img==2
                # Ejes
                plt.xlim(xs_adim[0],xs_adim[-1])
                plt.ylim(ys_adim[0],ys_adim[-1])

                # Solución
                sol_adim = solucion[:lim_plot_y] / (conc_ini)

                # Texto
                plt.text(x_texto, y_texto, texto,
                         bbox=dict(fill=True,facecolor="white",alpha=1.0),
                         va="center", ha="center")

                # Mapa de contornos
                paso_plot = 0.25
                niv_contourf = np.arange(c_inf,
                                         c_sup+paso_plot,
                                         paso_plot)

                niv_colorbar = np.arange(c_inf,
                                         c_sup+paso_plot,
                                         2*paso_plot)

                plt.contourf(xs_adim, ys_adim, sol_adim,
                             cmap="coolwarm", levels=niv_contourf)
                
                cbar = plt.colorbar(ticks=niv_colorbar)
                cbar.ax.set_ylabel(etiqueta)

            # Salida a archivo
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
