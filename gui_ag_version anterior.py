'''

import PySimpleGUI as sg
import pandas as pd
import os

def ejecutar_ga_y_mostrar_resultados(expresion):
    inicializar_variables(expresion)

    #decoder = decoder  # Asegúrate de definir correctamente 'decoder'
    obj_fun = dejong_OF

    ga_options = dict(
        num_eras=400, population_size=50, chromosome_length=len(VARIABLES) * 7,
        crossover_probability=0.90, mutation_probability=0.06
    )

    resultados = []
    for _ in range(10):  # Cambia  por el número de veces que deseas ejecutar el algoritmo
        plot_ga(obj_fun, decoder, expresion, min_or_max=MAX, ga_opts=ga_options, title="Función Objetivo (Maximización)")
        pre_result = [globals()[variable] for variable in VARIABLES]
        pre_result.append(eval(expresion))
        resultados.append(pre_result)

    return resultados, VARIABLES  # Devuelve también la lista de VARIABLES

def exportar_a_excel(resultados, variables, ruta_destino):
    variables.append("IS")
    df = pd.DataFrame(resultados, columns=variables)
    ruta_archivo = os.path.join(ruta_destino, "resultados_ga.xlsx")
    df.to_excel(r"C:\Users\midia\OneDrive\Escritorio\resultados_ga.xlsx", index=False)

    return ruta_archivo

def main():
    sg.theme("LightGrey1")

    layout = [
        [sg.Text("Introduce la expresión: "), sg.InputText(key="expresion")],
        [sg.Button("Ejecutar GA"), sg.Button("Salir")],
        [sg.Text("", size=(50, 1), key='result_text')],
    ]

    window = sg.Window("Ejecutar GA y Exportar Resultados", layout)

    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Salir"):
            break
        elif event == "Ejecutar GA":
            expresion = values["expresion"]
            resultados, variables = ejecutar_ga_y_mostrar_resultados(expresion)
            formatted_results = '\n'.join([' '.join(map(str, row)) for row in resultados])
            window['result_text'].update(formatted_results)

            # Obtener la ruta del escritorio
            escritorio = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

            # Exportar a Excel en el escritorio y mostrar mensaje emergente
            ruta_archivo = exportar_a_excel(resultados, variables, escritorio)
            sg.popup(f'Resultados exportados a {ruta_archivo}', title='Exportar a Excel')

    window.close()

if __name__ == '__main__':
    main()

'''