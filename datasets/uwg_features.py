_FEATURES = {
    'gender': [
        'Female',
    ],
    'body': [
        'BodyThin',
        'BodyNormal',
        'BodyFat',
    ],
    'viewpoint': [
        'Front',
        'Side',
        'Back',
    ],
    'age-group': [
        'AgeChild',
        'AgeYoungAdult',
        'AgeMediumAdult',
        'AgeOldAdult',
        'AgeSenior'
    ],
    'accessory': [
        'Backpack',
        'Bag',
        'HandTrunk',
        'No accessory',
        'OtherAttchment',
        'No carrying',
    ],
    'footware': [
        'Sandals',
        'Shoes',
        'LeatherShoes',
        'SportShoes',
        'Boots',
        'Sneaker',
        'ClothShoes',
    ],
    'apparent-action': [
        'Calling',
        'Talking',
        'Gathering',
        'Holding',
        'Pulling',
        'CarryingbyArm',
        'CarryingbyHand',
        'HoldObjectsInFront',
    ],
    'clothing': [
        'CasualWear',
        'Jacket',
        'FormalWear',
        'LongTrousers',
        'CarryingOther',
        'Dress',
        'LongCoat',
        'Trousers',
        'LongSleeve',
    ]
}

FEATURES = []

def get_features(args):
    global FEATURES

    if len(FEATURES):
        return FEATURES

    print('::: creating features list')

    for feature_category in args.appearance_categories:
        FEATURES.extend(_FEATURES[feature_category])

    return FEATURES
