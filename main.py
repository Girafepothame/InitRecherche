from functions import *



PATH = "dataset/dataset_caracters/01_Numeric_police12"



# Return all files from "path" directory (default all png files)
def char_paths(path = "dataset/dataset_caracters"):
    return glob.glob(path + "/**/*.png", recursive=True)


def main():
    tab = {}
    # instanciate dictionary with every letter and its png files by subdirectory
    for i in range(ord('a'), ord('z')+1):
        tab[chr(i)] = char_paths(PATH+"/"+chr(i))
        
            
    print(tab)
    for file in tab['a']:
        img = load(file)
    
    
if __name__ == "__main__":
    main()

