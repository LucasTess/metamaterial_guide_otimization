nome_original = "guide.fsp"
novas_letras = "log"

# O fatiamento [:-3] pega a string do início até o 3º caractere a partir do fim
# sem incluí-lo.
parte_inicial = nome_original[:-3]

# Concatena a parte inicial com a nova string
nome_modificado = parte_inicial + novas_letras

print(f"Nome original: {nome_original}")
print(f"Nome modificado: {nome_modificado}")