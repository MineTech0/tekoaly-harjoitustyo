
## Tiedot
Niilo Kurki Tietojenkäsittelytieteen kandiohjelma

Hallitut kielet: Java, Python, JavaScript

Dokumentaation kielenä käytetään suomea

# Määrittelydokumentti

## Aiheen kuvaus

Harjoitustyön aiheena on toteuttaa yksinkertainen konvoluutionneuroverkko, joka pystyy tunnistamaan musiikista kappaleen tanssilajin. Toteutuskielenä käytetään Pythonia. 

Toteutuksen testaamiseksi käytetään valmista konvoluutioneuroverkko toteutusta tensorflow kirjastosta. Tämä toteutus on koulutettu tunnistamaan kappaleen tanssilaji. Tämän jälkeen toteutetaan yksinkertainen konvoluutioneuroverkko, joka pystyy tunnistamaan kappaleen tanssilajin. Tämän jälkeen vertaillaan valmiin toteutuksen ja itse toteutetun konvoluutioneuroverkon suorituskykyä.

## Tavoitteet

Tavoitteena on toteuttaa yksinkertainen konvoluutioneuroverkko, joka pystyy tunnistamaan kappaleen tanssilajin. Tämän jälkeen vertaillaan valmiin toteutuksen ja itse toteutetun konvoluutioneuroverkon suorituskykyä.

## Toteutus

Malli koulutettaan kappaleilla, joista muodostetaan spektogrammit ja ne prosessoidaan niin, että korostavat rytmin eri osia. Nämä spektogrammit syötettään konvoluutio neuroverkolle, joka oppii tunnistamaan tanssilajin spektogrammin ominaisuuksien perusteella.