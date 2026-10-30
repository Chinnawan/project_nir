import os
import re
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR, PROJECT_ROOT
from src.consent import require_consent

RAW_DIR = os.path.join(PROJECT_ROOT, "images", "RawImages")
OUT_DIR = DATA_DIR

# ฐานข้อมูลจาก lookup.py (คัดลอกมาทั้ง 819 รายการ)
rawfile_map = {
    1: ("Russet", 1), 2: ("Healthy", 1), 3: ("Other", 1), 4: ("Bruise", 1),
    5: ("Hail_damage", 1), 6: ("Hail_damage_perf", 1), 7: ("Healthy", 2),
    8: ("Healthy", 3), 9: ("Stem", 1), 10: ("Stem", 2), 11: ("Calix", 1),
    12: ("Hail_damage", 2), 13: ("Healthy", 4), 14: ("Stem", 3), 15: ("Stem", 4),
    16: ("Calix", 2), 17: ("Calix", 3), 18: ("Calix", 4), 19: ("Russet", 2),
    20: ("Healthy", 5), 21: ("Healthy", 6), 22: ("Healthy", 7), 23: ("Stem", 5),
    24: ("Stem", 6), 25: ("Calix", 5), 26: ("Calix", 6), 27: ("Other", 2),
    28: ("Hail_damage_perf", 2), 29: ("Stem", 7), 30: ("Russet", 3),
    31: ("Scald", 1), 32: ("Healthy", 8), 33: ("Healthy", 9), 34: ("Stem", 8),
    35: ("Calix", 7), 36: ("Calix", 8), 37: ("Limb_rub", 1), 38: ("Healthy", 10),
    39: ("Russet", 4), 40: ("Russet", 5), 41: ("Frost_damage", 1), 42: ("Scald", 2),
    43: ("Healthy", 11), 44: ("Stem", 9), 45: ("Stem", 10), 46: ("Calix", 9),
    47: ("Calix", 10), 48: ("Hail_damage_perf", 3), 49: ("Hail_damage", 3),
    50: ("Healthy", 12), 51: ("Calix", 11), 52: ("Scald", 3), 53: ("Healthy", 13),
    54: ("Stem", 11), 55: ("Stem", 12), 56: ("Calix", 12), 57: ("Healthy", 14),
    58: ("Hail_damage", 4), 59: ("Stem", 13), 60: ("Calix", 13),
    61: ("Hail_damage_perf", 4), 62: ("Scald", 4), 63: ("Healthy", 15),
    64: ("Calix", 14), 65: ("Limb_rub", 2), 66: ("Other", 3), 67: ("Scald", 5),
    68: ("Other", 4), 69: ("Hail_damage_perf", 5), 70: ("Healthy", 16),
    71: ("Calix", 15), 72: ("Calix", 16), 73: ("Healthy", 17), 74: ("Limb_rub", 3),
    75: ("Russet", 6), 76: ("Hail_damage", 5), 77: ("Healthy", 18), 78: ("Stem", 14),
    79: ("Calix", 17), 80: ("Russet", 7), 81: ("Scald", 6), 82: ("Stem", 15),
    83: ("Calix", 18), 84: ("Hail_damage", 6), 85: ("Healthy", 19), 86: ("Stem", 16),
    87: ("Calix", 19), 88: ("Hail_damage", 7), 89: ("Limb_rub", 4), 90: ("Healthy", 20),
    91: ("Other", 5), 92: ("Calix", 20), 93: ("Other", 6), 94: ("Healthy", 21),
    95: ("Stem", 17), 96: ("Hail_damage_perf", 6), 97: ("Healthy", 22),
    98: ("Stem", 18), 99: ("Calix", 21), 100: ("Hail_damage_perf", 7),
    101: ("Calix", 22), 102: ("Frost_damage", 2), 103: ("Healthy", 23),
    104: ("Hail_damage_perf", 8), 105: ("Healthy", 24), 106: ("Calix", 23),
    107: ("Russet", 8), 108: ("Russet", 9), 109: ("Calix", 24),
    110: ("Frost_damage", 3), 111: ("Frost_damage", 4), 112: ("Rot", 1),
    113: ("Hail_damage_perf", 9), 114: ("Calix", 25), 115: ("Rot", 2),
    116: ("Other", 7), 117: ("Other", 8), 118: ("Stem", 19), 119: ("Healthy", 25),
    120: ("Russet", 10), 121: ("Healthy", 26), 122: ("Russet", 11),
    123: ("Other", 9), 124: ("Healthy", 27), 125: ("Stem", 20),
    126: ("Hail_damage_perf", 10), 127: ("Healthy", 28), 128: ("Healthy", 29),
    129: ("Calix", 26), 130: ("Scald", 7), 131: ("Other", 10), 132: ("Healthy", 30),
    133: ("Calix", 27), 134: ("Scald", 8), 135: ("Frost_damage", 5),
    136: ("Frost_damage", 6), 137: ("Hail_damage_perf", 11), 138: ("Healthy", 31),
    139: ("Russet", 12), 140: ("Other", 11), 141: ("Rot", 3), 142: ("Healthy", 32),
    143: ("Stem", 21), 144: ("Calix", 28), 145: ("Rot", 4), 146: ("Rot", 5),
    147: ("Rot", 6), 148: ("Frost_damage", 7), 149: ("Limb_rub", 5),
    150: ("Healthy", 33), 151: ("Hail_damage", 8), 152: ("Healthy", 34),
    153: ("Hail_damage_perf", 12), 154: ("Healthy", 35), 155: ("Calix", 29),
    156: ("Russet", 13), 157: ("Stem", 22), 158: ("Russet", 14), 159: ("Healthy", 36),
    160: ("Calix", 30), 161: ("Stem", 23), 162: ("Hail_damage_perf", 13),
    163: ("Healthy", 37), 164: ("Stem", 24), 165: ("Other", 12), 166: ("Healthy", 38),
    167: ("Stem", 25), 168: ("Rot", 7), 169: ("Hail_damage_perf", 14),
    170: ("Stem", 26), 171: ("Healthy", 39), 172: ("Scald", 9),
    173: ("Hail_damage", 9), 174: ("Hail_damage", 10), 175: ("Stem", 27),
    176: ("Calix", 31), 177: ("Healthy", 40), 178: ("Healthy", 41),
    179: ("Hail_damage_perf", 15), 180: ("Hail_damage_perf", 16),
    181: ("Healthy", 42), 182: ("Stem", 28), 183: ("Calix", 32), 184: ("Scald", 10),
    185: ("Healthy", 43), 186: ("Stem", 29), 187: ("Calix", 33),
    188: ("Frost_damage", 8), 189: ("Frost_damage", 9), 190: ("Frost_damage", 10),
    191: ("Hail_damage_perf", 17), 192: ("Other", 13), 193: ("Hail_damage_perf", 18),
    194: ("Hail_damage", 11), 195: ("Stem", 30), 196: ("Calix", 34),
    197: ("Healthy", 44), 198: ("Hail_damage_perf", 19), 199: ("Healthy", 45),
    200: ("Hail_damage_perf", 20), 201: ("Calix", 35), 202: ("Healthy", 46),
    203: ("Rot", 8), 204: ("Healthy", 47), 205: ("Calix", 36), 206: ("Scald", 11),
    207: ("Scald", 12), 208: ("Healthy", 48), 209: ("Calix", 37), 210: ("Russet", 15),
    211: ("Healthy", 49), 212: ("Stem", 31), 213: ("Russet", 16), 214: ("Healthy", 50),
    215: ("Calix", 38), 216: ("Russet", 17), 217: ("Healthy", 51), 218: ("Calix", 39),
    219: ("Hail_damage", 12), 220: ("Healthy", 52), 221: ("Calix", 40),
    222: ("Healthy", 53), 223: ("Hail_damage_perf", 21), 224: ("Bruise", 2),
    225: ("Calix", 41), 226: ("Other", 14), 227: ("Russet", 18), 228: ("Russet", 19),
    229: ("Calix", 42), 230: ("Healthy", 54), 231: ("Healthy", 55),
    232: ("Russet", 20), 233: ("Healthy", 56), 234: ("Stem", 32), 235: ("Calix", 43),
    236: ("Russet", 21), 237: ("Healthy", 57), 238: ("Healthy", 58),
    239: ("Stem", 33), 240: ("Calix", 44), 241: ("Other", 15), 242: ("Other", 16),
    243: ("Bruise", 3), 244: ("Healthy", 59), 245: ("Calix", 45), 246: ("Russet", 22),
    247: ("Russet", 23), 248: ("Russet", 24), 249: ("Hail_damage_perf", 22),
    250: ("Stem", 34), 251: ("Scald", 13), 252: ("Healthy", 60), 253: ("Healthy", 61),
    254: ("Stem", 35), 255: ("Hail_damage", 13), 256: ("Healthy", 62),
    257: ("Rot", 9), 258: ("Hail_damage", 14), 259: ("Healthy", 63),
    260: ("Stem", 36), 261: ("Calix", 46), 262: ("Healthy", 64), 263: ("Stem", 37),
    264: ("Calix", 47), 265: ("Russet", 25), 266: ("Healthy", 65),
    267: ("Russet", 26), 268: ("Stem", 38), 269: ("Hail_damage_perf", 23),
    270: ("Calix", 48), 271: ("Rot", 10), 272: ("Scald", 14), 273: ("Healthy", 66),
    274: ("Healthy", 67), 275: ("Stem", 39), 276: ("Russet", 27), 277: ("Healthy", 68),
    278: ("Healthy", 69), 279: ("Calix", 49), 280: ("Stem", 40), 281: ("Healthy", 70),
    282: ("Healthy", 71), 283: ("Stem", 41), 284: ("Calix", 50), 285: ("Healthy", 72),
    286: ("Healthy", 73), 287: ("Stem", 42), 288: ("Russet", 28), 289: ("Healthy", 74),
    290: ("Calix", 51), 291: ("Hail_damage_perf", 24), 292: ("Healthy", 75),
    293: ("Healthy", 76), 294: ("Stem", 43), 295: ("Calix", 52), 296: ("Other", 17),
    297: ("Healthy", 77), 298: ("Healthy", 78), 299: ("Stem", 44), 300: ("Rot", 11),
    301: ("Healthy", 79), 302: ("Healthy", 80), 303: ("Stem", 45), 304: ("Calix", 53),
    305: ("Russet", 29), 306: ("Healthy", 81), 307: ("Healthy", 82), 308: ("Stem", 46),
    309: ("Calix", 54), 310: ("Russet", 30), 311: ("Healthy", 83), 312: ("Stem", 47),
    313: ("Calix", 55), 314: ("Healthy", 84), 315: ("Hail_damage_perf", 25),
    316: ("Healthy", 85), 317: ("Stem", 48), 318: ("Calix", 56), 319: ("Limb_rub", 6),
    320: ("Russet", 31), 321: ("Healthy", 86), 322: ("Bruise", 4), 323: ("Calix", 57),
    324: ("Healthy", 87), 325: ("Hail_damage_perf", 26), 326: ("Russet", 32),
    327: ("Healthy", 88), 328: ("Healthy", 89), 329: ("Stem", 49), 330: ("Calix", 58),
    331: ("Russet", 33), 332: ("Rot", 12), 333: ("Healthy", 90), 334: ("Calix", 59),
    335: ("Russet", 34), 336: ("Healthy", 91), 337: ("Healthy", 92), 338: ("Stem", 50),
    339: ("Calix", 60), 340: ("Stem", 51), 341: ("Rot", 13), 342: ("Healthy", 93),
    343: ("Healthy", 94), 344: ("Stem", 52), 345: ("Calix", 61), 346: ("Bruise", 5),
    347: ("Healthy", 95), 348: ("Stem", 53), 349: ("Calix", 62), 350: ("Rot", 14),
    351: ("Healthy", 96), 352: ("Bruise", 6), 353: ("Healthy", 97), 354: ("Calix", 63),
    355: ("Stem", 54), 356: ("Rot", 15), 357: ("Healthy", 98), 358: ("Stem", 55),
    359: ("Calix", 64), 360: ("Healthy", 99), 361: ("Healthy", 100),
    362: ("Healthy", 101), 363: ("Stem", 56), 364: ("Stem", 57), 365: ("Calix", 65),
    366: ("Healthy", 102), 367: ("Hail_damage", 15), 368: ("Stem", 58),
    369: ("Healthy", 103), 370: ("Healthy", 104), 371: ("Stem", 59),
    372: ("Stem", 60), 373: ("Calix", 66), 374: ("Other", 18), 375: ("Healthy", 105),
    376: ("Calix", 67), 377: ("Healthy", 106), 378: ("Russet", 35), 379: ("Stem", 61),
    380: ("Calix", 68), 381: ("Healthy", 107), 382: ("Stem", 62), 383: ("Calix", 69),
    384: ("Healthy", 108), 385: ("Healthy", 109), 386: ("Calix", 70),
    387: ("Stem", 63), 388: ("Rot", 16), 389: ("Rot", 17), 390: ("Rot", 18),
    391: ("Russet", 36), 392: ("Stem", 64), 393: ("Healthy", 110), 394: ("Calix", 71),
    395: ("Rot", 19), 396: ("Healthy", 111), 397: ("Healthy", 112), 398: ("Stem", 65),
    399: ("Calix", 72), 400: ("Healthy", 113), 401: ("Healthy", 114),
    402: ("Calix", 73), 403: ("Stem", 66), 404: ("Healthy", 115),
    405: ("Healthy", 116), 406: ("Stem", 67), 407: ("Calix", 74),
    408: ("Healthy", 117), 409: ("Russet", 37), 410: ("Russet", 38),
    411: ("Calix", 75), 412: ("Healthy", 118), 413: ("Healthy", 119),
    414: ("Healthy", 120), 415: ("Calix", 76), 416: ("Healthy", 121),
    417: ("Calix", 77), 418: ("Healthy", 122), 419: ("Rot", 20), 420: ("Stem", 68),
    421: ("Healthy", 123), 422: ("Calix", 78), 423: ("Healthy", 124),
    424: ("Stem", 69), 425: ("Calix", 79), 426: ("Healthy", 125), 427: ("Stem", 70),
    428: ("Hail_damage_perf", 27), 429: ("Calix", 80), 430: ("Frost_damage", 11),
    431: ("Healthy", 126), 432: ("Healthy", 127), 433: ("Stem", 71),
    434: ("Calix", 81), 435: ("Russet", 39), 436: ("Scald", 15), 437: ("Stem", 72),
    438: ("Calix", 82), 439: ("Scald", 16), 440: ("Stem", 73), 441: ("Healthy", 128),
    442: ("Flesh_damage", 1), 443: ("Hail_damage", 16), 444: ("Healthy", 129),
    445: ("Stem", 74), 446: ("Calix", 83), 447: ("Healthy", 130),
    448: ("Healthy", 131), 449: ("Healthy", 132), 450: ("Healthy", 133),
    451: ("Stem", 75), 452: ("Calix", 84), 453: ("Healthy", 134),
    454: ("Healthy", 135), 455: ("Stem", 76), 456: ("Calix", 85),
    457: ("Other", 19), 458: ("Healthy", 136), 459: ("Healthy", 137),
    460: ("Rot", 21), 461: ("Stem", 77), 462: ("Calix", 86), 463: ("Healthy", 138),
    464: ("Healthy", 139), 465: ("Stem", 78), 466: ("Calix", 87),
    467: ("Flesh_damage", 2), 468: ("Limb_rub", 7), 469: ("Healthy", 140),
    470: ("Stem", 79), 471: ("Calix", 88), 472: ("Healthy", 141),
    473: ("Healthy", 142), 474: ("Calix", 89), 475: ("Hail_damage_perf", 28),
    476: ("Healthy", 143), 477: ("Calix", 90), 478: ("Stem", 80),
    479: ("Healthy", 144), 480: ("Healthy", 145), 481: ("Stem", 81),
    482: ("Bruise", 7), 483: ("Healthy", 146), 484: ("Healthy", 147),
    485: ("Stem", 82), 486: ("Stem", 83), 487: ("Calix", 91),
    488: ("Hail_damage_perf", 29), 489: ("Healthy", 148), 490: ("Healthy", 149),
    491: ("Stem", 84), 492: ("Stem", 85), 493: ("Calix", 92), 494: ("Healthy", 150),
    495: ("Healthy", 151), 496: ("Healthy", 152), 497: ("Stem", 86),
    498: ("Calix", 93), 499: ("Russet", 40), 500: ("Scald", 17),
    501: ("Healthy", 153), 502: ("Calix", 94), 503: ("Flesh_damage", 3),
    504: ("Healthy", 154), 505: ("Healthy", 155), 506: ("Healthy", 156),
    507: ("Calix", 95), 508: ("Stem", 87), 509: ("Stem", 88), 510: ("Healthy", 157),
    511: ("Healthy", 158), 512: ("Healthy", 159), 513: ("Stem", 89),
    514: ("Calix", 96), 515: ("Healthy", 160), 516: ("Healthy", 161),
    517: ("Healthy", 162), 518: ("Calix", 97), 519: ("Stem", 90),
    520: ("Healthy", 163), 521: ("Healthy", 164), 522: ("Healthy", 165),
    523: ("Calix", 98), 524: ("Stem", 91), 525: ("Stem", 92), 526: ("Healthy", 166),
    527: ("Healthy", 167), 528: ("Flesh_damage", 4), 529: ("Healthy", 168),
    530: ("Healthy", 169), 531: ("Healthy", 170), 532: ("Stem", 93),
    533: ("Calix", 99), 534: ("Calix", 100), 535: ("Stem", 94), 536: ("Healthy", 171),
    537: ("Rot", 22), 538: ("Healthy", 172), 539: ("Healthy", 173),
    540: ("Stem", 95), 541: ("Calix", 101), 542: ("Healthy", 174),
    543: ("Healthy", 175), 544: ("Stem", 96), 545: ("Calix", 102),
    546: ("Healthy", 176), 547: ("Healthy", 177), 548: ("Calix", 103),
    549: ("Stem", 97), 550: ("Bruise", 8), 551: ("Healthy", 178), 552: ("Stem", 98),
    553: ("Calix", 104), 554: ("Healthy", 179), 555: ("Healthy", 180),
    556: ("Stem", 99), 557: ("Stem", 100), 558: ("Healthy", 181),
    559: ("Healthy", 182), 560: ("Healthy", 183), 561: ("Stem", 101),
    562: ("Calix", 105), 563: ("Healthy", 184), 564: ("Healthy", 185),
    565: ("Healthy", 186), 566: ("Healthy", 187), 567: ("Stem", 102),
    568: ("Stem", 103), 569: ("Calix", 106), 570: ("Healthy", 188),
    571: ("Healthy", 189), 572: ("Healthy", 190), 573: ("Stem", 104),
    574: ("Hail_damage_perf", 30), 575: ("Healthy", 191), 576: ("Healthy", 192),
    577: ("Stem", 105), 578: ("Stem", 106), 579: ("Calix", 107),
    580: ("Healthy", 193), 581: ("Healthy", 194), 582: ("Healthy", 195),
    583: ("Healthy", 196), 584: ("Stem", 107), 585: ("Calix", 108),
    586: ("Healthy", 197), 587: ("Healthy", 198), 588: ("Healthy", 199),
    589: ("Calix", 109), 590: ("Stem", 108), 591: ("Healthy", 200),
    592: ("Healthy", 201), 593: ("Healthy", 202), 594: ("Stem", 109),
    595: ("Calix", 110), 596: ("Healthy", 203), 597: ("Healthy", 204),
    598: ("Calix", 111), 599: ("Stem", 110), 600: ("Healthy", 205),
    601: ("Healthy", 206), 602: ("Healthy", 207), 603: ("Stem", 111),
    604: ("Calix", 112), 605: ("Healthy", 208), 606: ("Healthy", 209),
    607: ("Healthy", 210), 608: ("Healthy", 211), 609: ("Calix", 113),
    610: ("Stem", 112), 611: ("Healthy", 212), 612: ("Healthy", 213),
    613: ("Healthy", 214), 614: ("Stem", 113), 615: ("Calix", 114),
    616: ("Healthy", 215), 617: ("Healthy", 216), 618: ("Stem", 114),
    619: ("Calix", 115), 620: ("Calix", 116), 621: ("Healthy", 217),
    622: ("Healthy", 218), 623: ("Healthy", 219), 624: ("Calix", 117),
    625: ("Healthy", 220), 626: ("Healthy", 221), 627: ("Healthy", 222),
    628: ("Healthy", 223), 629: ("Stem", 115), 630: ("Calix", 118),
    631: ("Healthy", 224), 632: ("Healthy", 225), 633: ("Calix", 119),
    634: ("Calix", 120), 635: ("Stem", 116), 636: ("Stem", 117),
    637: ("Healthy", 226), 638: ("Healthy", 227), 639: ("Healthy", 228),
    640: ("Healthy", 229), 641: ("Calix", 121), 642: ("Stem", 118),
    643: ("Healthy", 230), 644: ("Healthy", 231), 645: ("Calix", 122),
    646: ("Stem", 119), 647: ("Bruise", 9), 648: ("Healthy", 232),
    649: ("Hail_damage_perf", 31), 650: ("Stem", 120), 651: ("Calix", 123),
    652: ("Healthy", 233), 653: ("Healthy", 234), 654: ("Healthy", 235),
    655: ("Stem", 121), 656: ("Calix", 124), 657: ("Healthy", 236),
    658: ("Healthy", 237), 659: ("Healthy", 238), 660: ("Stem", 122),
    661: ("Stem", 123), 662: ("Calix", 125), 663: ("Calix", 126),
    664: ("Healthy", 239), 665: ("Russet", 41), 666: ("Healthy", 240),
    667: ("Healthy", 241), 668: ("Other", 20), 669: ("Stem", 124),
    670: ("Healthy", 242), 671: ("Healthy", 243), 672: ("Stem", 125),
    673: ("Calix", 127), 674: ("Stem", 126), 675: ("Healthy", 244),
    676: ("Healthy", 245), 677: ("Healthy", 246), 678: ("Calix", 128),
    679: ("Stem", 127), 680: ("Healthy", 247), 681: ("Healthy", 248),
    682: ("Calix", 129), 683: ("Stem", 128), 684: ("Healthy", 249),
    685: ("Healthy", 250), 686: ("Calix", 130), 687: ("Healthy", 251),
    688: ("Healthy", 252), 689: ("Healthy", 253), 690: ("Healthy", 254),
    691: ("Stem", 129), 692: ("Stem", 130), 693: ("Calix", 131),
    694: ("Calix", 132), 695: ("Healthy", 255), 696: ("Healthy", 256),
    697: ("Healthy", 257), 698: ("Stem", 131), 699: ("Calix", 133),
    700: ("Calix", 134), 701: ("Healthy", 258), 702: ("Healthy", 259),
    703: ("Healthy", 260), 704: ("Stem", 132), 705: ("Healthy", 261),
    706: ("Healthy", 262), 707: ("Healthy", 263), 708: ("Stem", 133),
    709: ("Stem", 134), 710: ("Calix", 135), 711: ("Healthy", 264),
    712: ("Healthy", 265), 713: ("Stem", 135), 714: ("Calix", 136),
    715: ("Rot", 23), 716: ("Healthy", 266), 717: ("Calix", 137),
    718: ("Healthy", 267), 719: ("Stem", 136), 720: ("Calix", 138),
    721: ("Healthy", 268), 722: ("Healthy", 269), 723: ("Healthy", 270),
    724: ("Stem", 137), 725: ("Stem", 138), 726: ("Calix", 139),
    727: ("Healthy", 271), 728: ("Healthy", 272), 729: ("Healthy", 273),
    730: ("Stem", 139), 731: ("Stem", 140), 732: ("Calix", 140),
    733: ("Healthy", 274), 734: ("Healthy", 275), 735: ("Stem", 141),
    736: ("Stem", 142), 737: ("Healthy", 276), 738: ("Healthy", 277),
    739: ("Healthy", 278), 740: ("Stem", 143), 741: ("Stem", 144),
    742: ("Calix", 141), 743: ("Calix", 142), 744: ("Russet", 42),
    745: ("Stem", 145), 746: ("Stem", 146), 747: ("Calix", 143),
    748: ("Calix", 144), 749: ("Healthy", 279), 750: ("Healthy", 280),
    751: ("Stem", 147), 752: ("Stem", 148), 753: ("Calix", 145),
    754: ("Bruise", 10), 755: ("Bruise", 11), 756: ("Bruise", 12),
    757: ("Bruise", 13), 758: ("Bruise", 14), 759: ("Bruise", 15),
    760: ("Bruise", 16), 761: ("Bruise", 17), 762: ("Bruise", 18),
    763: ("Bruise", 19), 764: ("Bruise", 20), 765: ("Bruise", 21),
    766: ("Bruise", 22), 767: ("Bruise", 23), 768: ("Bruise", 24),
    769: ("Bruise", 25), 770: ("Bruise", 26), 771: ("Bruise", 27),
    772: ("Bruise", 28), 773: ("Bruise", 29), 774: ("Bruise", 30),
    775: ("Bruise", 31), 776: ("Bruise", 32), 777: ("Bruise", 33),
    778: ("Bruise", 34), 779: ("Bruise", 35), 780: ("Bruise", 36),
    781: ("Bruise", 37), 782: ("Bruise", 38), 783: ("Bruise", 39),
    784: ("Bruise", 40), 785: ("Bruise", 41), 786: ("Bruise", 42),
    787: ("Bruise", 43), 788: ("Bruise", 44), 789: ("Bruise", 45),
    790: ("Bruise", 46), 791: ("Bruise", 47), 792: ("Bruise", 48),
    793: ("Bruise", 49), 794: ("Bruise", 50), 795: ("Bruise", 51),
    796: ("Bruise", 52), 797: ("Bruise", 53), 798: ("Bruise", 54),
    799: ("Bruise", 55), 800: ("Flesh_damage", 5), 801: ("Flesh_damage", 6),
    802: ("Flesh_damage", 7), 803: ("Flesh_damage", 8), 804: ("Flesh_damage", 9),
    805: ("Flesh_damage", 10), 806: ("Flesh_damage", 11), 807: ("Flesh_damage", 12),
    808: ("Flesh_damage", 13), 809: ("Flesh_damage", 14), 810: ("Flesh_damage", 15),
    811: ("Flesh_damage", 16), 812: ("Flesh_damage", 17), 813: ("Flesh_damage", 18),
    814: ("Flesh_damage", 19), 815: ("Flesh_damage", 20), 816: ("Flesh_damage", 21),
    817: ("Flesh_damage", 22), 818: ("Flesh_damage", 23), 819: ("Flesh_damage", 24),
}

plan_b = {
    "Healthy": "Healthy",
    "Stem": "Healthy",
    "Calix": "Healthy",
    "Bruise": "Internal_Defect", # ปรับให้ไม่มีเว้นวรรค ป้องกันบั๊กตอนอ่าน Path
    "Rot": "Internal_Defect",
    "Flesh_damage": "Internal_Defect",
    "Russet": "Surface_Defect",
    "Scald": "Surface_Defect",
    "Hail_damage": "Surface_Defect",
    "Hail_damage_perf": "Surface_Defect",
    "Frost_damage": "Surface_Defect",
    "Limb_rub": "Surface_Defect",
    "Other": "Surface_Defect",
}

def get_label(filename):
    match = re.search(r'(\d+)', filename)
    if not match: return None
    num = int(match.group(1))
    if num not in rawfile_map: return None
    
    defect_type, _ = rawfile_map[num]
    return plan_b.get(defect_type, "Unknown")

def setup_directories():
    if os.path.exists(OUT_DIR) and os.listdir(OUT_DIR):
        raise FileExistsError(f"{OUT_DIR} already contains data; back it up and use an empty output directory")
    for split in ["train", "val", "test"]:
        for cls in ["Healthy", "Internal_Defect", "Surface_Defect"]:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

def crop_and_resize(quadrant):
    # quadrant ต้นฉบับคือ 512 (H) x 640 (W)
    # Center Crop ให้เป็นสี่เหลี่ยมจัตุรัส 512x512 โดยตัดขอบซ้ายขวาออกฝั่งละ 64 พิกเซล
    cropped = quadrant[:, 64:576]
    return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)

def process_and_stack(filepath):
    require_consent()
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # หั่น 4 Quadrant ตาม Layout: IR (บนซ้าย), GR (บนขวา), BL (ล่างซ้าย), RE (ล่างขวา)
    ir_quad = img[0:512, 0:640]
    gr_quad = img[0:512, 640:1280]
    bl_quad = img[512:1024, 0:640]
    re_quad = img[512:1024, 640:1280]

    # ทำ Center Crop และ Resize
    ir_224 = crop_and_resize(ir_quad)
    gr_224 = crop_and_resize(gr_quad)
    bl_224 = crop_and_resize(bl_quad)
    re_224 = crop_and_resize(re_quad)

    # วาง Stack ตามลำดับความยาวคลื่น BL(450), GR(500), RE(750), IR(800) -> [0, 1, 2, 3]
    return np.stack([bl_224, gr_224, re_224, ir_224], axis=-1)

def main():
    require_consent()
    print("กำลังจัดเตรียมโครงสร้างโฟลเดอร์...")
    setup_directories()
    
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.tif', '.png', '.jpg'))]
    valid_files, labels = [], []
    
    for f in files:
        label = get_label(f)
        if label and label != "Unknown":
            valid_files.append(f)
            labels.append(label)

    total_files = len(valid_files)
    print(f"✅ ตรวจพบไฟล์ที่จับคู่ Label สำเร็จ: {total_files} ไฟล์")
    if total_files != 819:
        print("⚠️ ระวัง: จำนวนไฟล์ไม่ครบ 819 ตามเป้าหมาย")

    # Stratified Split 70 : 15 : 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        valid_files, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    for split_name, (file_list, label_list) in splits.items():
        print(f"🚀 กำลังประมวลผลและหั่นภาพชุด {split_name} ({len(file_list)} ไฟล์)...")
        for fname, cls in zip(file_list, label_list):
            raw_path = os.path.join(RAW_DIR, fname)
            data_cube = process_and_stack(raw_path)
            
            if data_cube is not None:
                save_name = fname.replace(".tif", ".npy").replace(".jpg", ".npy")
                save_path = os.path.join(OUT_DIR, split_name, cls, save_name)
                np.save(save_path, data_cube)

    print("🎯 Data Pipeline เสร็จสมบูรณ์ ข้อมูลพร้อมสำหรับการเทรน!")

if __name__ == "__main__":
    main()