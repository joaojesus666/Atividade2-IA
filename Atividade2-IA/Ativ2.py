# importaçao das Bibliotecas
import pandas as pd     # Manipulaçao de dados
# Importando funçao train_test_split
from sklearn.model_selection import train_test_split
# Funçao que faz a Classificaçao, trabalha com arvores de decisao
from sklearn.ensemble import ExtraTreesClassifier
# importando 'confusion_matrix' para testa modelos por classe
from sklearn.metrics import confusion_matrix

# COVID-19 Sintomas
'''
Sintomas mais comuns:
– tosse
– febre
– falta de ar progressiva
– chiado

Sintomas menos comuns:
- dores e desconfortos
- dor de garganta
- diarreia
- conjuntivite
- dor de cabeça
- perda de paladar ou olfato
- erupção cutânea na pele ou descoloração dos dedos das mãos ou dos pés

sintomas graves:
- pneumonia, 
- síndrome respiratória aguda grave 
- insuficiência renal 
- morte.

Pessoas de Risco
- idosos, 
- crianças 
- pacientes oncológicos, 
- portadores de imunodeficiências, 
- doenças respiratórias

'''

# variavel DadosDF recebe DataFrame
DadosDF = pd.read_csv('INFLUD21-21-02-2022', sep=';')


# ***       Tratamento de Dados         ***

# axis = 0 linha , axis = 1 Coluna
# Retirando as colunas desnecessarias
# B) Remoção de variáveis relacionadas ao óbito (data do óbito e número da declaração de óbito)

Dados2DF = DadosDF.drop(['DT_NOTIFIC','SEM_NOT','DT_SIN_PRI','SEM_PRI','SG_UF_NOT','ID_REGIONA','CO_REGIONA','ID_MUNICIP','CO_MUN_NOT','ID_UNIDADE','CO_UNI_NOT','DT_NASC','TP_IDADE', 'COD_IDADE','CS_GESTANT','CS_RACA','CS_ETINIA','CS_ESCOL_N','ID_PAIS','CO_PAIS','SG_UF','ID_RG_RESI','CO_RG_RESI','ID_MN_RESI','CO_MUN_RES','CS_ZONA','AVE_SUINO','SIND_DOWN','NEUROLOGIC','VACINA','DT_UT_DOSE','MAE_VAC','DT_VAC_MAE','M_AMAMENTA','DT_DOSEUNI','DT_1_DOSE','DT_2_DOSE','ANTIVIRAL','TP_ANTIVIR','OUT_ANTIV','DT_ANTIVIR','DT_INTERNA','SG_UF_INTE','ID_RG_INTE','CO_RG_INTE','ID_MN_INTE','CO_MU_INTE','DT_ENTUTI','DT_SAIDUTI','RAIOX_RES','RAIOX_OUT','DT_RAIOX','DT_COLETA','TP_AMOSTRA','OUT_AMOST','DT_EVOLUCA','DT_ENCERRA','DT_DIGITA','DT_VGM','DT_RT_VGM','PAC_COCBO','PAC_DSCBO','OUT_ANIM','DT_TOMO','DT_CO_SOR','DT_RES','SURTO_SG','AN_ADENO','AN_OUTRO','DS_AN_OUT','TP_AM_SOR','SOR_OUT','TP_SOR','OUT_SOR','RES_IGG','RES_IGM','RES_IGA','HISTO_VGM','PAIS_VGM','CO_PS_VGM','LO_PS_VGM','PCR_SARS2','DOR_ABD','FADIGA','PERD_OLFT','PERD_PALA','TOMO_RES','TOMO_OUT','TP_TES_AN','DT_RES_AN','RES_AN','POS_AN_FLU','TP_FLU_AN','POS_AN_OUT','AN_SARS2','AN_VSR','AN_PARA1','AN_PARA2','AN_PARA3','AMOSTRA','PCR_RESUL','DT_PCR','POS_PCRFLU','TP_FLU_PCR','PCR_FLUASU','FLUASU_OUT','PCR_FLUBLI','FLUBLI_OUT','POS_PCROUT','PCR_VSR','PCR_PARA1','PCR_PARA2','PCR_PARA3','PCR_PARA4','PCR_ADENO','PCR_METAP','PCR_BOCA','PCR_RINO','PCR_OUTRO','DS_PCR_OUT','CLASSI_OUT','OBES_IMC','OUT_MORBI','MORB_DESC','SUPORT_VEN','PUERPERA','CS_SEXO','OUTRO_SIN','VOMITO','OUTRO_DES'], axis=1)

# Poderia ter pegado somente as colunas que eu queria e colocar dentro de uma lista , porem achei melhor pelo metodo de exclusao , para analisar uma maior quantidade de dados.

# A) Remoção de casos de SRAG não diagnosticados como COVID-19;

# ['CLASSI_FIN'] Diagnostico Final do Caso
# 1-SRAG por influenza
# 2-SRAG por outro vírus respiratório
# 3-SRAG por outro agente etiológico, qual:
# 4-SRAG não especificado
# 5-SRAG por COVID-19

Dados2DF = Dados2DF.loc[Dados2DF['CLASSI_FIN'] < 5]


# tratando variaveis Na
# laço 'for' substitui dados faltantes por 0
# e substitui dados 'S' em 1 e 'N' em 0 para equilibrar a analise

for val in Dados2DF:
    Dados2DF[val] = Dados2DF[val].fillna(0)
    Dados2DF[val] = Dados2DF[val].replace('S',1)
    Dados2DF[val] = Dados2DF[val].replace('N',0)


# Filtrando a tabela 
# como a coluna [EVOLUÇAO] trata , cura e obitos 
# 1 = cura e 2 = obito
Dados2DF = Dados2DF.loc[DadosDF['EVOLUCAO'] < 3]

# C) Seleção e tratamento de variáveis;

# separando as variaveis entre preditoras e alvo
# Variavel y é nosso alvo ['EVOLUCAO'] , pois obtem os casos de obitos e curas
y = Dados2DF['EVOLUCAO']
# drop exclui e 'axis' modifica so a coluna , deicxando a variavel x com o restante dos dados para as analises
x = Dados2DF.drop('EVOLUCAO' , axis=1)

# D ) Treino de classificador, para classificar entre “óbito por COVID-19” e “cura” (informação da coluna “EVOLUCAO”);

# Essa funçao separa as variaveis entre teste e treino 
# teste_size = 0.3 igual a 30% sera para teste e 70% pra treino
# Excolhi esse valor para teste pois aparenta um bom resultado
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.3)

# criaçao do modelo
model = ExtraTreesClassifier()

# fit() aplica o algoritmo machine learning nos dados (treino)
model.fit(x_treino, y_treino)

# E) Avaliar classificador e reportar a acurácia geral e por classe em conjunto de teste.

# Imprimindo rsultado
# Funçao score compara os algoritmos de teste(que sao os dados reais)
# Vai tentar fazer a previsao em cima de cada variavel

resultado = model.score(x_teste, y_teste)
# %round() para deixar apenas 2 casas depois da virgula
print('Acuracia: %.2f' % round(resultado,2),'% \n')

# Acuracia por classe
y_predi = model.predict(x_teste)
confusao = confusion_matrix(y_teste, y_predi)

# resultado :
result = confusao[0][0] / (confusao[0][0] + confusao[1][0] ) #Cura
result1 = confusao[0][1] / (confusao[0][1] + confusao[1][1] )  #Obito

# %round() para deixar apenas 2 casas depois da virgula
print('Curados: %.2f' %round(result,2) ,'% \n')
print('Obitos: %.2f' %round(result1,2) ,'% \n')