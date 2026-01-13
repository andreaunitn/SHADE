import os

res_1 = "337x600"
res_2 = "1012x1800"
res_3 = "1687x3000"

var1 = 0

print("da dove partire?")
var1 = input()

var1 = int(var1)

for i in range(0, 30, 1):

    if i == 0:
        old_name = "/Users/andreatomasoni/Downloads/Immagini/WhatsApp Image 2021-12-10 at 09.53.26.jpeg"
    else:
        old_name = "/Users/andreatomasoni/Downloads/Immagini/WhatsApp Image 2021-12-10 at 09.53.26 (" + str(i) + ").jpeg"

    if i%3 == 2:
        if var1 < 10:
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(var1) + "-" + str(res_3) + ".jpeg"
        else:     
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(var1) + "-" + str(res_3) + ".jpeg"
        
    elif i%3 == 1:
        if var1 < 10:
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(var1) + "-" + str(res_2) + ".jpeg"
        else:
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(var1) + "-" + str(res_2) + ".jpeg"

    else:
        if var1 < 10:
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-00" + str(var1) + "-" + str(res_1) + ".jpeg"
        else:
            new_name = new_name = "/Users/andreatomasoni/Downloads/Immagini/original-0" + str(var1) + "-" + str(res_1) + ".jpeg"

    os.rename(old_name, new_name)

    if i%3 == 2:
        var1 = var1 + 1