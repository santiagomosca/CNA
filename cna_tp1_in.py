#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1

archivo: 'v2_in_tp1.py'

Función que lee el archivo de input de las simulaciones y asigna valores
a las variables requeridas por el programa 'v2_main_tp1.py'.
"""

import re
import sys
from numpy import float

#**** FUNCIONES DEL PROGRAMA ****#

def datos_input(archivo_input, comentario='#'):
    """
    Función que lee un archivo de entrada para recoger las variables
    requeridas por el programa principal.

    Informa si alguno de los valores de input no es válido.
    """

    # Nombres de variables numéricas que usa
    # el programa 'v2_main_tp1.py'
    nom_var_num_prog = [
                        'x_ini',
                        'x_fin',
                        'y_ini',
                        'y_fin',
                        'vel',
                        'h',
                        'f',
                        'e_l',
                        'e_t',
                        'desc_cont',
                        'c_dec',
                        't_total',
                        'dx',
                        'dy',
                        'dt',
                        'theta',
                       ]

    # Nombres de variables alfabéticas que usa
    # el programa 'v2_main_tp1.py'
    nom_var_alfa_prog = [
                         'cb_x_ini',
                         'cb_x_fin',
                         'cb_y_ini',
                         'cb_y_fin',
                         'upwinding'
                        ]

    # Conversión a mayúsculas de los nombres de variables en
    # 'nom_var_prog'. Variables a ser leídas de 'archivo_input'
    nom_var_num_in = [var.upper() for var in nom_var_num_prog]
    nom_var_alfa_in = [var.upper() for var in nom_var_alfa_prog]    

    # Lectura de líneas de 'archivo_input' y captura de datos
    dict_valores_num = {}
    dict_valores_alfa = {}
    with open(archivo_input, 'r') as a_in:
        # Lista para comprobar que las variables no han sido
        # especificadas más de una vez o estén faltantes
        variables_encontradas = []
        for linea in a_in:
            if not linea.startswith(comentario):
                # Adición de variables numéricas
                for var_num in nom_var_num_in:
                    if re.search(r'\b' + var_num + r'\b', linea):
                        variables_encontradas.append(var_num)
                        try:
                            val_num = linea.split('=')[-1].strip()
                            dict_valores_num.update({var_num:val_num})
                        except:
                            print("Variable requerida " +\
                                  "'{}' ".format(var_num) +\
                                  "sin especificar")
                            sys.exit(1)
                # Adición de variables alfabéticas
                for var_alfa in nom_var_alfa_in:
                    if re.search(r'\b' + var_alfa + r'\b', linea):
                        variables_encontradas.append(var_alfa)
                        try:
                            val_alfa = linea.split('=')[-1].strip()
                            dict_valores_alfa.update({var_alfa:val_alfa})
                        except:
                            print("Variable requerida " +\
                                  "'{}' ".format(var_alfa) +\
                                  "sin especificar")
                            sys.exit(1)

        # Comprobación de que las variables requeridas no estén
        # faltantes o hayan sido especificadas más de una vez
        for var_num in nom_var_num_in:
            if variables_encontradas.count(var_num) < 1:
                print("Variable requerida {} faltante".format(var_num))
                sys.exit(1)
            elif variables_encontradas.count(var_num) > 1:
                print("Variable requerida {} ".format(var_num) +\
                      "especificada más de una vez")
                sys.exit(1)
            else:
                pass

        for var_alfa in nom_var_alfa_in:
            if variables_encontradas.count(var_alfa) < 1:
                print("Variable requerida {} faltante".format(var_alfa))
                sys.exit(1)
            elif variables_encontradas.count(var_alfa) > 1:
                print("Variable requerida {} ".format(var_alfa) +\
                      "especificada más de una vez")
                sys.exit(1)
            else:
                pass

    # Verifica si las variables alfabéticas cumplen
    # con el tipo necesario
    for key in dict_valores_alfa:
        # Verificación de variable alfabética
        if not dict_valores_alfa[key].isalpha():
            print("La variable requerida para {}".format(key)
                  + " debe ser del tipo alfabética")
            sys.exit(1)

        # Verificación de las condiciones de borde
        elif key in nom_var_alfa_in[:-1]:
            if dict_valores_alfa[key] not in ('DIR','NEU'):
                print(dict_valores_alfa)
                print("Tipo de condición de borde mal especificado")
                print("La condición debe ser 'DIR' o 'NEU'")
                sys.exit(1)
            else:
                pass

        # Verificación de 'upwinding'
        else:
            if dict_valores_alfa[key] not in ('SI','NO'):
                print("Selección de 'upwinding' mal especificada")
                print("La selección debe ser 'SI' o 'NO'")
                sys.exit(1)
            else:
                pass

    # Conversión a flotante de las variables numéricas
    for key, val in dict_valores_num.items():
        if val.isalpha():
            print("La variable {} debe ser numérica".format(key))
            sys.exit(1)
        else:
            dict_valores_num[key] = float(val)

    # Verificación del valor de theta
    if dict_valores_num['THETA'] > 1.0 or \
       dict_valores_num['THETA'] < 0.0:
        print("Error: THETA debe variar entre 0 y 1")
        sys.exit(1)
    else:
        pass

    # Construcción del diccionario final para el programa 'v2_main_tp1.py'
    dicc_prog = {**dict_valores_alfa, **dict_valores_num}
    
    return dicc_prog

#**** FIN PROGRAMA ****#
