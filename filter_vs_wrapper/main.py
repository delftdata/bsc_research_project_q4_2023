from Methods.Chi2.internet_ads import internet_ads_chi2
from Methods.ANOVA.arrhythmia import arrhythmia_anova
from Methods.ForwardSelection.crop_fs import crop_fs
from Methods.BackwardElimination.character_font_images import character_font_images_be


def main():
    internet_ads_chi2()

    arrhythmia_anova()

    crop_fs()

    character_font_images_be()


if __name__ == '__main__':
    main()
