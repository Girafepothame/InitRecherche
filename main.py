from functions import *



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
    car_tab = []
    # test_tab = []
    code_tab = []
    
    min_tab = []
    for char in char_tab:
        skel = skeletonize(char, method = "lee")
        
        ## Affichage / Debug
        car_tab.append(draw_minutia(minutia_extraction(skel), skel, (255, 0, 0)))
        ret, min = smoothing(skel, minutia_extraction(skel), 12)
        min_tab.append(draw_minutia(min, skel, (255, 0, 0)))
        
        ## Freeman encoding
        cache = []
        code = freeman_encode(skel, cache)
        code_tab.append(code)
        # test = draw_minutia(cache, skel, (0, 255, 0))
        # test_tab.append(test)
        # Post treating the freeman code
    
    for code in code_tab:
        print(code)
    
    affiche_tab(char_tab)
    affiche_tab(car_tab)
    affiche_tab(min_tab)
    # affiche_tab(test_tab)
    
    
if __name__ == "__main__":
    main()

