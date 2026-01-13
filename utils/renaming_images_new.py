import os

res_1 = "337x600"
res_2 = "1012x1800"
res_3 = "1687x3000"

i = 0

print("Dove partire?")
inizio = input()
inizio_int = int(inizio)

dir_name = '/Users/andreatomasoni/Downloads/Immagini'
list_of_image = sorted(os.listdir(dir_name))

for image in list_of_image:

    if i%3 == 1:
        if inizio_int < 10:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(inizio_int) + "-" + str(res_2) + ".jpeg"
        else:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(inizio_int) + "-" + str(res_2) + ".jpeg"

    elif i%3 == 2:
        if inizio_int < 10:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(inizio_int) + "-" + str(res_3) + ".jpeg"
        else:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(inizio_int) + "-" + str(res_3) + ".jpeg"
        
    else:
        if inizio_int < 10:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(inizio_int) + "-" + str(res_1) + ".jpeg"
        else:
            new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(inizio_int) + "-" + str(res_1) + ".jpeg"

    i = i + 1

    if i%3 == 0:
        inizio_int = inizio_int + 1