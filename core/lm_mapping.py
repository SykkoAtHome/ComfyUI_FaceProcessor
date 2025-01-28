class LandmarkMappings:
    # Each tuple represents (mediapipe_point_id, dlib_point_id)
    landmarks_pairs = {
        "left_eye": [
            (133, 40),
            (33, 37)
        ],
        "right_eye": [
            (362, 43),
            (263, 46)
        ],
        "mouth": [
            (0, 52),
            (17, 58),
            (61, 49),
            (291, 55),
            (13, 63),
            (14, 67),
            (78, 61),
            (308, 65)
        ],
        "node_edge": [
            (6, 28),
            (4, 31)
        ],
        "oval": [
            (152, 9),
            (127, 1),
            (356, 17),
            (172, 5),
            (397, 13)
        ]
    }
