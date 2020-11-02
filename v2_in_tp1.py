#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-

"""
Cálculo numérico avanzado, 2020, FIUBA.

Grupo 4: Santiago Mosca, Santiago Pérez Raiden, Cristhian Zárate Evers.

Trabajo Práctico n.° 1

Programa 'v2_in_tp1.py'

Función que lee el archivo de input de las simulaciones y asigna valores
a las variables requeridas por el programa 'v2_main_tp1.py'.
"""

import re
from numpy import float

#**** FUNCIONES DEL PROGRAMA ****#

def datos_input(archivo_input, comentario='#'):
    """
    Función que lee un archivo de entrada para recoger las variables
    requeridas por el programa principal.

    Informa si alguno de los valores de input no es válido.
    """

    # Nombres de variables que usa el programa 'v2_main_tp1.py'
    nom_var_prog = [
                    'x_ini',
                    'x_fin',
                    'y_ini',
                    'y_fin',
                    'vel',
                    'h',
                    'f',
                    'e_l',
                    'e_t'
                    'desc_cont',
                    'c_dec',
                    'dx',
                    'dy',
                    'dt',
                    'cb_x_ini',
                    'cb_x_fin',
                    'cb_y_ini',
                    'cb_y_fin',
                    'theta',
                    'upwinding'
                   ]

    # Conversión a mayúsculas de los nombres de variables en
    # 'nom_var_prog'. Variables a ser leídas de 'archivo_input'
    nom_var_in = [var.upper() for var in nom_var_prog]

    valores = []
    # Lectura de líneas de 'archivo_input' y captura de datos
    with open(archivo_input, 'r') as a_in:
        for linea in a_in:
            if not linea.startswith(comentario):
                for var in nom_var_in:
                    if re.search(r'\b' + var + r'\b', linea):
                        valores.append(linea.split('=')[-1].split()[0])

    # Conversión a 'float' de los valores numéricos
    # Verifica si str es alfabética. Los números flotantes devuelven
    # 'falso' por ser alfanuméricos
    for cont, val in enumerate(valores):
        if not val.isalpha():
            valores[cont] = float(val)
        else:
            pass
 
    # Construcción del diccionario final para el programa 'v2_main_tp1.py'
    dicc_prog = dict(zip(nom_var_prog, valores))

    return dicc_prog

#**** FIN PROGRAMA ****#
