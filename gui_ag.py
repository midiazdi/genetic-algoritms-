import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['figure.figsize'] = (18.0, 8.0)
import random
from typing import List, Union, Tuple, Callable
import pandas as pd
import re
import PySimpleGUI as sg

#PLOBLACION
################################################################################
def coin(prob:float) -> bool:
    """
    Performs a biased coin toss.
    :param prob: [0 ≤ float ≤ 1]
    :returns: [bool] True with probability `prob` and otherwise False
    """
    # random.random() yields a float between 0 and 1
    return random.random() < prob


def random_string(length:int) -> str:
    """
    :param length: [int] length of random string
    :returns: [string] random string consisting of "0" and "1"
    """
    return ''.join('0' if coin(0.5) else '1' for _ in range(length))


def generate_random_population(number:int, length:int) -> List[str]:
    """
    This function is used to generate the first population.
    This implementation ensure that chromosomes in the initial population
    are uniformly pseudo-random!

    :param number: [int] number of strings to return
    :param length: [int] length of the strings to return
    :returns: List[str] list of random binary strings
    """
    return [random_string(length) for _ in range(number)]


#REPRODUCCION
################################################################################
MIN = 0
MAX = 1


def reproduction(population:List[str], fitness_func:Callable, min_or_max:int =MAX) -> List[str]:
    """
    Produces a new population from biased roulette reproduction of the
    given population.
    :param population: [List[str]]
    :param fitness_func: [function: number > 0]
    :param min_or_max: {MIN, MAX}
    :returns: [List[str]]
    """
    # First, we define the probability density (roulette weights) for each
    # member in our given population.

    min_fitness = min(fitness_func(m) for m in population)

    def compute_weight(m):
        """
        Subroutine which computes the weight of the biased roulette, which is agnostic of the fitness function. In particular, it will invert the fitness value if we are seeking a minimum. Member `m` has weight that is commensurate with its distance from the member with lowest fitness in the population.
        :param m: [str] member
        """
        fitness = fitness_func(m)

        if min_or_max == MAX:
            return fitness - min_fitness + 1

        elif min_or_max == MIN:
            return 1 / (fitness - min_fitness + 1)

    # Here we normalize the weights to be proportions of the total weighting
    weights = [(m, compute_weight(m)) for m in population]
    total_weights = sum(w for m, w in weights)
    pdf = [(m, w/total_weights) for m, w in weights]

    # Now we pick members for the new population.
    # We pick the same number of members as the provided population.
    new_population = []
    for i in range(len(population)):
        rand = random.random()
        cumul = 0
        for member, end_interval in pdf:
            cumul += end_interval
            if rand <= cumul:
                new_population.append(member)
                break # generate next member

    return new_population


#COMBINACION
################################################################################
def crossover(string1, string2, index):
    """ Performs crossover on two strings at given index """
    head1, tail1 = string1[:index], string1[index:]
    head2, tail2 = string2[:index], string2[index:]
    return head1+tail2, head2+tail1


def population_crossover(population:List[str], crossover_probability:float) -> List[str]:
    """
    Performs crossover on an entire population.
    :param population: List[str]
    :param crossover_probability: [0 ≤ float ≤ 1]
        chance that any pair will be crossed over
    :returns: List[str]
        new population with possibly some members crossed over
    """
    pairs = []
    new_population = []
    while len(population) > 1:
        pairs.append((population.pop(), population.pop()))
    if len(population) == 1:
        new_population.append(population.pop())

    for s1, s2 in pairs:
        if not coin(crossover_probability):
            # don't perform crossover, just add the original pair
            new_population += [s1, s2]
            continue
        idx = random.randint(1, len(s1)-1) # select crossover index
        new_s1, new_s2 = crossover(s1, s2, idx)
        new_population.append(new_s1)
        new_population.append(new_s2)
    return new_population



#MUTACION
################################################################################
def mutation(string:str, probability:float)->str:
    """
    :param string: the binary string to mutate
    :param probability: [0 ≤ float ≤ 1]
        the probability of any character being flipped
    :returns: [str]
        just the input string, possibly with some bits flipped
    """

    flipped = lambda x: '1' if x == '0' else '0'
    chars = (flipped(char) if coin(probability) else char for char in string)
    return ''.join(chars)

def mutate_population(population:List[str], prob:float)->List[str]:
    """
    :param population: [List[str]]
        population of binary strings
    :returns: [List[str]]
        just the input population with some members possibly mutated
    """
    return [mutation(m, prob) for m in population]


#FUNCION para agrupar las variables internas de cada dimesion
#############################################################
def agrupar_letras(lista):
    letras_agrupadas = {}
    
    for elemento in lista:
        for caracter in elemento:
            if caracter.islower():
                if caracter in letras_agrupadas:
                    letras_agrupadas[caracter].append(elemento)
                else:
                    letras_agrupadas[caracter] = [elemento]
    
    return list(letras_agrupadas.values())
def extrar_valor_variables(variable_str):
    ind = VARIABLES[VARIABLES.index(variable_str)]
    return globals()[ind]
#FUNCION para ordenar la lista con las mayusculas de primeras
################################################################################
def ordenar_lista(lista):
    return sorted(lista, key=lambda x: (x.lower(), x))

#FUNCION para obtener los indices de las letras mayusculas
################################################################################
def indices_mayusculas(lista):
    return [i for i, elemento in enumerate(lista) if elemento.isupper()]
#FUNCION INICIALIZADORA DE VARIABLES
################################################################################
def inicializar_variables(expresion):
    # Encuentra todas las variables en la expresión
    global VARIABLES
    VARIABLES = re.findall(r'([a-zA-Z]\w*)', expresion)

    # Crea las variables globales con valor inicial 0
    for variable in VARIABLES:
        globals()[variable] = 0

    return VARIABLES

#GENERACION PRINCIPAL
################################################################################
def run_genetic_algorithm(obj_fun,condicion, decoder,expresion,
                          min_or_max=MAX, num_eras=100,
                          population_size=20, chromosome_length=12,
                          crossover_probability=0.4,mutation_probability=0.005):

    # define fitness function (decode string, then feed to the OF)
    fitness = lambda coding: obj_fun(decoder(coding,expresion),condicion)

    # initialize population
    population = generate_random_population(number=population_size,length=chromosome_length)
    # data collection
    populations = [population] # initialize with first population

    # SGA loop
    for i in range(num_eras):
        population = reproduction(population, fitness, min_or_max)
        population = population_crossover(population, crossover_probability)
        population = mutate_population(population, mutation_probability)
        populations.append(population) # data collection

    return populations


#FUNCIONES
#######################################################################################
def dejong_OF(expresion,condicion):
    #Penalización si las variables toman determinados valores
    if condicion != "-":
        if eval(condicion):
            return eval(expresion)
        else: return 0
    else: return eval(expresion)
        



    #return (0.17*x + 0.11*w)*y + (0.63*u + 0.12*t)*v + (0.68*r + 0.96*q)*s + 0.16*p
    #return (0.92*x + 0.34*w)*y + (0.89*u + 0.14*t)*v + (0.63*r + 0.93*q)*s + 0.54*p
    #return (0.49*x + 0.10*w)*y + (1*u + 0.24*t)*v + (0.87*q)*s + 1*p

    #return (0.46*x + 0.10*w)*y + (0.63*u + 0.01*t)*v + (0.68*r + 0.96*q)*s + 1*p
    #return (0.93*x + 0.40*w)*y + (0.53*u + 0.06*t)*v + (0.63*r + 0.93*q)*s + 0.28*p
    #return (0.70*x + 0.09*w)*y + (1*u + 0.19*t)*v + (0.50*r + 0.88*q)*s + 1*p
    #return (0.94*x + 0.18*w)*y + (0.11*u + 0.21*t)*v + (0.47*r + 0.87*q)*s + 0.10*p

#funcion para obtener la lista que contiene a la variable x
def obtener_lista(lista_de_listas, x):
    for sublista in lista_de_listas:
        if x in sublista:
            return sublista
    return None 


#DECODIFICADORES
#######################################################################################


def internal_decoder(variable_eva,variables_interna)->int:

    for var in variables_interna:
      if var != variable_eva:
        sum += globals()[var] 

    if sum == 0:
        return variable_eva
    else:
        return variable_eva/sum

def external_decoder(ind):
  sum=0
  for var in VARIABLES:
    if var.isupper() and var != ind :
      sum +=  globals()[var]
  return globals()[ind]/sum

#Normalizador
########################################################################
def normalizar(valores):
    for variable in VARIABLES:
        # Extraer el índice del valor actual
        indice = VARIABLES.index(variable)
        
        # Normalizar variables mayúsculas
        if variable.isupper():
            variables_mayusculas = [v for v in VARIABLES if v.isupper()]
            globals()[variable] = valores[indice] / sum(valores[VARIABLES.index(v)] for v in variables_mayusculas)
        # Normalizar variables minúsculas
        else:
            clase_variable = variable[:-1]  # Obtener la clase de la variable (t1 -> t)
            variables_minusculas = [v for v in VARIABLES if v.islower() and v[:-1] == clase_variable]
            suma = sum(valores[VARIABLES.index(v)] for v in variables_minusculas)
            if suma == 0:
                globals()[variable] = valores[indice]
            else:
                globals()[variable] = valores[indice] / suma



#Decoder
##################################################################################
def decoder(coding:str,expresion:str)->List[int]:
    cadenas = [coding[i:i+7] for i in range(0, len(coding), 7)]
    # use binary x and y as interval multiplier
    numeros = []
    for cadena in cadenas:
        numeros.append(int(cadena, 2))
    
    normalizar(numeros)
    return expresion


#GRAFICADORES
#######################################################################################
def plot_ga(obj_fun,condicion, decoder,expresion, ax=None, ga_opts=None, min_or_max=MIN,
            title="Genetic Algorithm Evolution", legend=True):
    if ga_opts is None:
        ga_opts = {}

    ga_opts['min_or_max'] = min_or_max
    # run SGA
    populations = run_genetic_algorithm(obj_fun,condicion, decoder,expresion, **ga_opts.copy())

    # define fitness func
    fitness = lambda c: obj_fun(decoder(c,expresion),condicion)

    # Find the "global optimum" of all the chromosomes we looked at.
    # A better term for this chromosome is "best individual".
    all_chromosomes = {c for pop in populations for c in pop}
    optimizer = min if min_or_max == MIN else max
    global_optimum = optimizer(all_chromosomes, key=fitness)
    fittest_fitness = fitness(global_optimum)

    # Print the optimum to the console
    #print("Global optimum:", global_optimum)
    #print("Fitness:", fittest_fitness)
    names = ['w1-tech','w2-tech','w2-eco','w1-eco','w1-env','w2-env','W1','W2','W3','W4']


    decoded_op = decoder(global_optimum,expresion)
    #for valor, name in zip(decoded_op,names):
    #    numero_formateado = f"{valor:.2f}"
    #    print(name, ':', numero_formateado)


    return decoded_op
    # Start plotting
    # Define the data ranges
    x_axis = range(len(populations))
    fitnesses = [[fitness(m) for m in population] for population in populations]

    mins = [min(f) for f in fitnesses]
    maxs = [max(f) for f in fitnesses]
    avgs = [sum(f)/len(f) for f in fitnesses]
    optima = [(it, fittest_fitness) for it, pop in enumerate(populations) if fittest_fitness in map(fitness, pop)]

    x_optima, y_optima = zip(*optima) # unzip pairs into two sequences
    print('Puntos optimos: ',x_optima,y_optima)
    if ax is None: # if no plotting axes are provided
        # define a set of axes
        fig, ax = plt.subplots(1)

    """
    # do the plotting
    l_mins, l_maxs, l_avgs = ax.plot(x_axis, mins, 'r--', maxs, 'b--', avgs, 'g-')
    scatter_ceil = ax.scatter(x_optima, y_optima, c='purple')
    # create a legend
    """
    # Crear la figura y los ejes
    #fig, ax = plt.subplots()

    # Graficar las curvas mínimas, máximas y promedio
    l_mins, = ax.plot(x_axis, mins, 'r--', label='Mínimos')
    l_maxs, = ax.plot(x_axis, maxs, 'b--', label='Máximos')
    l_avgs, = ax.plot(x_axis, avgs, 'g-', label='Promedio')

    # Rellenar el área entre las curvas mínimas y máximas
    ax.fill_between(x_axis, mins, maxs, color='gray', alpha=0.5)
    scatter_ceil = ax.scatter(x_optima, y_optima, c='purple')
    """
    if legend:
        plt.legend(
            (l_mins, l_maxs, l_avgs, scatter_ceil),
            ("min fitness", "max  fitness", "avg  fitness", "optimo global"),
            loc="best",
        )
    """
    # set parameters for the axes
    ax.set_xlim(0, len(populations))
    ax.set_ylim(0, int(max(maxs)+1))
    ax.set_title(title)
    ax.set_xlabel("era")
    ax.set_ylabel("fitness")
    #plt.show()
    return ax


#LLAMADO
#######################################################################################

    



import PySimpleGUI as sg
import pandas as pd
import os

def ejecutar_ga_y_mostrar_resultados(expresion, condicion, num_iteraciones):
    inicializar_variables(expresion)

    obj_fun = dejong_OF

    ga_options = dict(
        num_eras=num_iteraciones, population_size=50, chromosome_length=len(VARIABLES) * 7,
        crossover_probability=0.90, mutation_probability=0.06
    )

    resultados = []
    for _ in range(num_iteraciones):
        plot_ga(obj_fun, condicion, decoder, expresion, min_or_max=MAX, ga_opts=ga_options, title="Función Objetivo (Maximización)")
        pre_result = [globals()[variable] for variable in VARIABLES]

        resultado_adicional = eval(expresion)
        pre_result.append(resultado_adicional)

        resultados.append(pre_result)

    return resultados, VARIABLES

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
        [sg.Text("Introduce la condición: "), sg.InputText(key="condicion")],
        [sg.Text("Número de iteraciones: "), sg.InputText(key="num_iteraciones")],
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
            condicion = values["condicion"]
            num_iteraciones = int(values["num_iteraciones"])  # Convertir a entero

            resultados, variables = ejecutar_ga_y_mostrar_resultados(expresion, condicion, num_iteraciones)
            formatted_results = '\n'.join([' '.join(map(str, row)) for row in resultados])
            window['result_text'].update(formatted_results)

            escritorio = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            ruta_archivo = exportar_a_excel(resultados, variables, escritorio)
            sg.popup(f'Resultados exportados a {ruta_archivo}', title='Exportar a Excel')

    window.close()

if __name__ == '__main__':
    main()
