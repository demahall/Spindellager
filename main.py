from pipeline import Pipeline

if __name__ == '__main__':
    image_path = "Bilder/Bild_Beispiel.tif"  # Path relative to the current directory (Spindellager Projekt folder)
    pipeline = Pipeline(image_path)
    pipeline.run()
