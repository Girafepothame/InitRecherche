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
        
    char_tab = img_tab(pathtab, 'a')
    affiche_tab(char_tab)
    bin_tab = []
    inv_tab = []
    skel_tab = []
    for char in char_tab:
        # bin = cv2.threshold(char, 127, 255, cv2.THRESH_BINARY)
        # bin_tab.append(bin)
        char = invert_image(char)
        inv_tab.append(char)
        char = skeletonize(char, method = "lee")
        skel_tab.append(char)
        
    affiche_tab(inv_tab)
    affiche_tab(skel_tab)
    
    
if __name__ == "__main__":
    main()

