from functions import *



PATH = "dataset/dataset_caracters/01_Numeric_police12"

def affiche_tab(tab):
    titles = [i+1 for i in range(10)]
    for i in range(len(tab)):
        plt.subplot(2, 5, i+1),plt.imshow(tab[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    

def main():
    pathtab = {}
    # instanciate dictionary with every letter and its png files by subdirectory
    for i in range(ord('a'), ord('z')+1):
        pathtab[chr(i)] = char_paths(PATH+"/"+chr(i))
        
    char_tab = img_tab(pathtab, 'j')
    # affiche_tab(char_tab)
    inv_tab = []
    skel_tab = []
    min_tab = []
    for char in char_tab:
        inv = invert_image(char)
        inv_tab.append(inv)
        skel = skeletonize(inv, method = "lee")
        skel_tab.append(skel)
        min = minutia_extraction(skel)
        print(min)
        print(findFirst(skel))
        min = draw_minutia(min, skel)
        min_tab.append(min)        
    
    # freeman_encode(skel_tab[0])
    affiche_tab(skel_tab)
    affiche_tab(min_tab)
    
    
if __name__ == "__main__":
    main()

