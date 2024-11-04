import face_recognition
from PIL import Image, ImageDraw


def detecter_visages(path_img):
    image = face_recognition.load_image_file(path_img)
    face_locations = face_recognition.face_locations(image)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        draw.rectangle(((left, top), (right, bottom)), outline = "red", width = 2)

    return pil_image, len(face_locations)


def reconnaitre_visage(path_img_connue, path_img_test):
    image_connue = face_recognition.load_image_file(path_img_connue)
    image_test = face_recognition.load_image_file(path_img_test)

    encodage_connu = face_recognition.face_encodings(image_connue)[0]
    encodages_test = face_recognition.face_encodings(image_test)

    if not encodages_test:
        return "Aucun visage détecté dans l'image test"

    resultats = face_recognition.compare_faces([encodage_connu], encodages_test[0])
    distance = face_recognition.face_distance([encodage_connu], encodages_test[0])

    return {
        "match": bool(resultats[0]),
        "similarite": f"{(1 - distance[0]) * 100:.2f}%"
    }


if __name__ == "__main__":
    image, nb_visages = detecter_visages("images/anniversaire.png")
    print(f"Nombre de visages détectés : {nb_visages}")
    image.show()

    resultat = reconnaitre_visage("images/anniversaire.png", "images/paul_0.png")
    if isinstance(resultat, dict):
        print(f"Match trouvé : {resultat['match']}")
        print(f"Similarité : {resultat['similarite']}")
    else:
        print(resultat)
