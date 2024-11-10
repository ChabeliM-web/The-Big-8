"""
This code creates a DataFrame from a dictionary containing information about various schools in Gauteng. The data includes columns for school 
ID, school name, school phase (Primary or Secondary), the address of the school, the education district, the performance percentage for 2023, 
and contact details (email and phone). Once the DataFrame is populated with the information, it is saved as a CSV file named 
'gauteng_schools.csv'. The CSV file is generated without the index, providing a structured and accessible format of the school data that can
 be further analyzed or utilized.

"""

import pandas as pd

#data for the schools
data = {
    "schoo_ID": [
        700910011, 700400393, 700121210, 700350561, 700915064, 700400277, 700320291,
        700231522, 700231530, 700211276, 700320937, 700152033, 700400391, 700910158,
        700910169, 700221474, 700350595, 700211300, 700320366, 700270645, 700220574,
        700111740, 700152058, 700400031, 700400179, 700400178, 700400180, 700400212,
        700251306, 700400010, 700400423, 700400424, 700400149, 700910274, 700910276,
        700321927, 700914251, 700251363, 700320457, 700160028, 700220608, 700330837,
        700340570, 700260745
    ],
    "NAME OF SCHOOL": [
        "ADAM MASEBE SECONDARY SCHOOL", "ALBERTINA SISULU PRIMARY SCHOOL", "ALTMONT TECHNICAL HIGH SCHOOL",
        "ASSER MALOKA SECONDARY SCHOOL", "BACHANA MOKWENA PRIMARY SCHOOL", "BARCELONA PRIMARY SCHOOL",
        "BARRAGE PRIMARY FARM SCHOOL", "BATHABILE PRIMARY FARM SCHOOL", "BATHOKWA PRIMARY SCHOOL",
        "BEKEKAYO PRIMARY FARM SCHOOL", "BEVERLY HILLS SECONDARY SCHOOL", "BLAIR ATHOLL PRIMARY FARM SCHOOL",
        "BLUE EAGLE HIGH SCHOOL", "BOITSHEPO SECONDARY SCHOOL", "BOKAMOSO HIGH SCHOOL",
        "BONA LESEDI SECONDARY SCHOOL", "BONGANI PRIMARY FARM SCHOOL", "BOSCHKOP PRIMARY FARM SCHOOL",
        "BOTLEHADI PRIMARY SCHOOL", "BRANDVLEI PRIMARY FARM SCHOOL", "BULA-DIKGORO PRIMARY SCHOOL",
        "BUYANI PRIMARY SCHOOL", "CARTER PRIMARY SCHOOL", "CHIEF BAMBATA PRIMARY SCHOOL",
        "COSMO CITY JUNIOR PRIMARY SCHOOL", "COSMO CITY PRIMARY NO 1 SCHOOL", "COSMO CITY SECONDARY SCHOOL",
        "COSMO CITY WEST PRIMARY SCHOOL", "DIE POORT PRIMARY FARM SCHOOL", "DIEPSLOOT COMBINED SCHOOL",
        "DIEPSLOOT PRIMARY SCHOOL", "DIEPSLOOT SECONDARY SCHOOL NO. 2", "DIEPSLOOT WEST SECONDARY SCHOOL",
        "DIKAGO DINTLE PRIMARY SCHOOL", "DIKGAKOLOGO PRIMARY SCHOOL", "DINOKENG PRIMARY FARM SCHOOL",
        "DR. MOTSUENYANE SECONDARY SCHOOL", "DURBAN DEEP PRIMARY SCHOOL", "ED MASHABANE SECONDARY SCHOOL",
        "EKURHULENI PRIMARY SCHOOL", "EMASANGWENE PRIMARY SCHOOL", "EMMANUEL PRIMARY SCHOOL",
        "ENCOCHOYINI PRIMARY SCHOOL", "ENDULWENI PRIMARY SCHOOL"
    ],
    "SCHOOL PHASE": [
        "SEC", "PRI", "SEC", "SEC", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "SEC", "PRI", "SEC", "SEC",
        "SEC", "SEC", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "PRI", "SEC", "PRI",
        "PRI", "SEC", "PRI", "SEC", "SEC", "PRI", "PRI", "PRI", "SEC", "PRI", "SEC", "PRI", "PRI", "PRI", 
        "PRI", "PRI"
    ],
    "ADDRESS OF SCHOOL": [
        "110, BLOCK A, SEKAMPANENG, TEMBA, TEMBA, 0407", "1250, SIBUSISO, KINGSWAY, BENONI, BENONI, 1501",
        "24936, CNR ALEKHINE & STANTON RD, PROTEA SOUTH, SOWETO, JOHANNESBURG, 1818",
        "2544, MANDELA & TAMBO, BLUEGUMVIEW, DUDUZA, NIGEL, 1496",
        "2201, MAMASIYANOKA, GA-RANKUWA VIEW, GA-RANKUWA, PRETORIA, 0208",
        "22640, NGUNGUNYANE AVENUE, BARCELONA, ETWATWA, BENONI, 1519",
        "577, KAALPLAATS, BARRAGE, VANDERBIJLPARK, JOHANNESBURG, 1900",
        "11653, LINDANI STREET, OLIEVENHOUTBOSCH, CENTURION, PRETORIA, 0175",
        "1, LEPHORA STREET, SAULSVILLE, PRETORIA, PRETORIA, 0125",
        "25, OLD PRETORIA ROAD BAPSFONTEIN, BAPSFONTEIN, BENONI, BENONI, 1510",
        "2854, FLORIDA STREET, BEVERLY HILLS, EVATON WEST, VANDERBIJLPARK, 1984",
        "512, MALIBONGWE ROAD, LANSERIA, KRUGERSDORP, JOHANNESBURG, 1748",
        "1291, TANZANIA, COSMO CITY, RANDBURG, JOHANNESBURG, 2188",
        "1468, BABELEGI STR, MAROKOLONG VILLAGE, HAMMANSKRAAL, PRETORIA, 0400",
        "5506, NEWSTAND, STINKWATER, HAMMANSKRAAL, PRETORIA, 0407",
        "415, LEKGANYANE STREET, MAHUBE VALLEY, MAMELODI, PRETORIA, 0122",
        "29, ERMELO ROAD & STRYDPAN, KWAZENZELE COMMUNITY TRUST, ENDICOTT, SPRINGS, 1574",
        "43, PLOT NO 43, RAYTON, PRETORIA, PRETORIA, 1001",
        "1461, WARD ROAD, EVATON, EVATON, EVATON, 1984",
        "55, VENTERSDORP/RUSTENBURG ROAD, BRANDVLEI, RANDFONTEIN, RANDFONTEIN, 1760",
        "27400, RAMMUPUDU, MAMELODI EAST, MAMELODI, PRETORIA, 0122",
        "2022, CENTRAL STREET, FINETOWN, JOHANNESBURG, 1828",
        "44, 4TH AVENUE, ALEXANDRA, SANDTON, JOHANNESBURG, 2090",
        "3466, PANSY STREET, EVATON WEST, EVATON, VANDERBIJLPARK, 1984",
        "2632, ANGOLA AVENUE, COSMO CITY, RANDBURG, JOHANNESBURG, 2188",
        "2631, ANGOLA AVENUE, COSMO CITY, RANDBURG, JOHANNESBURG, 2194",
        "2322, ANGOLA AND KENYA, COSMO CITY, RANDBURG, JOHANNESBURG, 2162",
        "3760, EQUADOR DRIVE, COSMO CITY, RANDBURG, JOHANNESBURG, 2188",
        "23, HAARTEBEESFONTEIN, HEKPOORT, MOGALE CITY, MOGALE CITY, 1790",
        "3292, CNR UBUNTU AND LAPENG STR, DIEPSLOOT, JOHANNESBURG, JOHANNESBURG, 2189",
        "1, NGONYAMA & HUMBULANI, DIEPSLOOT, FOURWAYS, JOHANNESBURG, 2189",
        "000, CNR HUMBULANI & NGONYAMA, DIEPSLOOT EXT 5, DIEPSLOOT, RANDBURG, 2189",
        "1797, CHIEF LANGALIBALELE, EXTENTION 7, DIEPSLOOT, JOHANNESBURG, 2089",
        "1060, WINTERVELDT, WINTERVELDT, PRETORIA, 0198",
        "1336, KLIPPAN, WINTERVELDT, WINTERVELDT, PRETORIA, 0198",
        "581, MARLBANK, DRIEFONTEIN, VANDERBJILPARK, VANDERBJILPARK, 1900",
        "1571, MOLAPO MASEKO ROAD, WINTERVELT, PRETORIA, PRETORIA, 0198",
        "12, ENOCH SONTONGA STREET, DURBAN DEEP, RODEPOORT, JOHANNESBURG, 1725",
        "29, CNR FREDERICK & SELBORNE ROAD, SMALL FARM, VANDERBIJLPARK, VANDERBIJLPARK, 1911",
        "110, CNR RONDEBOSCH & RIVERSIDE RD, MORNINGSIDE, SANDTON, JOHANNESBURG, 2196",
        "43, RAZOR STREET, KATLEHONG SOUTH, KATLEHONG, KATLEHONG, 1432",
        "01, RHULANI SECTION, TEMBISA, TEMBISA, TEMBISA, 1632",
        "20, RHINO STR, KATHORUS, ALBERTON, ALBERTON, 1452",
        "3456, EBAN STR, SOWETO, JOHANNESBURG, JOHANNESBURG, 1868"
    ],
    "EDUCATION DISTRICT": [
        "GAUTENG NORTH", "EKURHULENI NORTH", "JOHANNESBURG WEST", "EKURHULENI EAST", "TSHWANE WEST",
        "EKURHULENI NORTH", "JOHANNESBURG WEST", "TSHWANE SOUTH", "TSHWANE SOUTH", "EKURHULENI NORTH",
        "SEDIBENG WEST", "JOHANNESBURG WEST", "JOHANNESBURG NORTH", "TSHWANE WEST", "TSHWANE WEST",
        "TSHWANE SOUTH", "GAUTENG EAST", "TSHWANE SOUTH", "SEDIBENG WEST", "WEST RAND", "TSHWANE SOUTH",
        "JOHANNESBURG SOUTH", "JOHANNESBURG EAST", "SEDIBENG WEST", "JOHANNESBURG NORTH",
        "JOHANNESBURG NORTH", "JOHANNESBURG NORTH", "JOHANNESBURG NORTH", "WEST RAND", "JOHANNESBURG NORTH",
        "JOHANNESBURG NORTH", "JOHANNESBURG NORTH", "JOHANNESBURG NORTH", "TSHWANE NORTH", "TSHWANE NORTH",
        "JOHANNESBURG WEST", "TSHWANE NORTH", "JOHANNESBURG WEST", "SEDIBENG WEST", "JOHANNESBURG EAST",
        "EKURHULENI NORTH", "SEDIBENG WEST", "GAUTENG WEST", "GAUTENG EAST"
    ],
    "PERFORMANCE PERCENTAGE 2023": [
        75, 82, 66, 73, 88, 70, 91, 64, 77, 72, 85, 78, 90, 67, 74, 81, 68, 79, 69, 87, 83, 76, 88, 80, 65,
        74, 92, 75, 81, 69, 84, 86, 79, 82, 90, 78, 66, 72, 88, 80, 93, 89, 71, 68
    ],
    "Contact Details": [
    {"Email": "adammasebe.secondary@gmail.com", "Phone": "+27 10 123 4567"},
    {"Email": "albertinasisulu.primary@gmail.com", "Phone": "+27 10 234 5678"},
    {"Email": "altmont.techhigh@gmail.com", "Phone": "+27 10 345 6789"},
    {"Email": "assermaloka.secondary@gmail.com", "Phone": "+27 10 456 7890"},
    {"Email": "bachanamokwenap.primary@gmail.com", "Phone": "+27 10 567 8901"},
    {"Email": "barcelona.primary@gmail.com", "Phone": "+27 10 678 9012"},
    {"Email": "barrage.primaryfarm@gmail.com", "Phone": "+27 10 789 0123"},
    {"Email": "bathabile.primaryfarm@gmail.com", "Phone": "+27 10 890 1234"},
    {"Email": "bathokwa.primary@gmail.com", "Phone": "+27 10 901 2345"},
    {"Email": "bekekayo.primaryfarm@gmail.com", "Phone": "+27 10 123 3456"},
    {"Email": "beverlyhills.secondary@gmail.com", "Phone": "+27 10 234 4567"},
    {"Email": "blairatholl.primaryfarm@gmail.com", "Phone": "+27 10 345 5678"},
    {"Email": "blueeagle.high@gmail.com", "Phone": "+27 10 456 6789"},
    {"Email": "boitswepo.secondary@gmail.com", "Phone": "+27 10 567 7890"},
    {"Email": "bokamoso.high@gmail.com", "Phone": "+27 10 678 8901"},
    {"Email": "bonalesedi.secondary@gmail.com", "Phone": "+27 10 789 9012"},
    {"Email": "bongani.primaryfarm@gmail.com", "Phone": "+27 10 890 0123"},
    {"Email": "boschkop.primaryfarm@gmail.com", "Phone": "+27 10 901 1234"},
    {"Email": "botlehadi.primary@gmail.com", "Phone": "+27 10 123 4568"},
    {"Email": "brandvlei.primaryfarm@gmail.com", "Phone": "+27 10 234 5679"},
    {"Email": "buladikgoro.primary@gmail.com", "Phone": "+27 10 345 6780"},
    {"Email": "buyani.primary@gmail.com", "Phone": "+27 10 456 7891"},
    {"Email": "carter.primary@gmail.com", "Phone": "+27 10 567 8902"},
    {"Email": "chiefbambata.primary@gmail.com", "Phone": "+27 10 678 9013"},
    {"Email": "cosmocity.juniorprimary@gmail.com", "Phone": "+27 10 789 0124"},
    {"Email": "cosmocity.primary1@gmail.com", "Phone": "+27 10 890 1235"},
    {"Email": "cosmocity.secondary@gmail.com", "Phone": "+27 10 901 2346"},
    {"Email": "cosmocitywest.primary@gmail.com", "Phone": "+27 10 123 3457"},
    {"Email": "diepoort.primaryfarm@gmail.com", "Phone": "+27 10 234 4568"},
    {"Email": "diepsloot.combined@gmail.com", "Phone": "+27 10 345 5679"},
    {"Email": "diepsloot.primary@gmail.com", "Phone": "+27 10 456 6780"},
    {"Email": "diepslootsecondary2@gmail.com", "Phone": "+27 10 567 7891"},
    {"Email": "diepslootwest.secondary@gmail.com", "Phone": "+27 10 678 8902"},
    {"Email": "dikagodintle.primary@gmail.com", "Phone": "+27 10 789 9013"},
    {"Email": "dikgakologo.primary@gmail.com", "Phone": "+27 10 890 0124"},
    {"Email": "dinokeng.primaryfarm@gmail.com", "Phone": "+27 10 901 1235"},
    {"Email": "drmotsuenyane.secondary@gmail.com", "Phone": "+27 10 123 4569"},
    {"Email": "durbandeep.primary@gmail.com", "Phone": "+27 10 234 5670"},
    {"Email": "edmashabane.secondary@gmail.com", "Phone": "+27 10 345 6781"},
    {"Email": "ekurhuleni.primary@gmail.com", "Phone": "+27 10 456 7892"},
    {"Email": "emasangwene.primary@gmail.com", "Phone": "+27 10 567 8903"},
    {"Email": "emmanuel.primary@gmail.com", "Phone": "+27 10 678 9014"},
    {"Email": "encochoyini.primary@gmail.com", "Phone": "+27 10 789 0125"},
    {"Email": "endulweni.primary@gmail.com", "Phone": "+27 10 890 1236"}
]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('gauteng_schools.csv', index=False)

print("Data saved to 'gauteng_schools.csv'")