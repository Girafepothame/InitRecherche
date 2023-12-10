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
        
    char_tab = img_tab(pathtab, 'c')
    
    min_tab = []
    for char in char_tab:
        inv = invert_image(char)
        skel = skeletonize(inv, method = "lee")
        min = smoothing(minutia_extraction(skel), 15)
        min_tab.append(draw_minutia(min, skel))
        cache = []
        print(freeman_encode(skel, cache))
    
    
    affiche_tab(min_tab)
    
    
if __name__ == "__main__":
    main()

