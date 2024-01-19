while True:
    try:
        # Demander à l'utilisateur d'entrer un nombre
        nombre = int(input("Entrez un nombre (ou 'q' pour quitter) : "))

        # Vérifier si le nombre est pair ou impair
        if nombre % 2 == 0:
            print("Nombre pair")
        else:
            print("Nombre impair")

    except ValueError:
        # Gérer le cas où l'entrée n'est pas un nombre entier
        user_input = input("Voulez-vous quitter ? (oui/non) : ").lower()
        if user_input == 'oui' or user_input == 'q':
            break
        else:
            print("Entrée invalide. Veuillez entrer un nombre entier.")
