Svrha dataseta
Dataset se koristi za obučavanje modela detekcije objekata koji prepoznaje:
- Pravilno parkirana vozila
- Nepravilno parkirana vozila (van parking oznaka, preko linija, parkiranje na zabranjenim mjestima)
- Parking oznake i granice

Karakteristike dataseta
- Tip podataka: Slike iz ptičije perspektive
- Izvor: Originalne fotografije makete parking prostora
- Broj slika: 400
- Anotacija: Manuelno označeno u Label Studio-u
- Format anotacija: YOLO

Izrada dataseta
Slike su snimljene na fizičkoj maketi parking prostora koja sadrži:
- Realističke parking oznake i razgraničenja
- Modele automobila u različitim pozicijama
- Simulaciju pravilnih i nepravilnih parking scenarija

Sve slike su fotografisane iz perspektive koja simulira pogled drona sa različitih visina i uglova kako bi se model pripremio za realne uslove.
Model gleda slike iz pričije perspektive i kada uoči prekršaj ili nepravilnost šalje komandu za približavanje detektovanom automobili.
Obzirom da u ovom projektu simuliramo ponašanje drona, nema stvarnog približavanja već samo nova, bliža slika za registraciju tablica. 