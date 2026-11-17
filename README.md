# wind-generator-forecasting
Weather diagnosis and prediction system for wind power generation

# Weather Diagnosis & Prediction

Kunlik ob-havo ma'lumotidan **bugungi ob-havoni diagnoz qiladi** (`snow`, `rain`, `windy`, `cloudy`, `sunny`)
va **ertangi kunni bashorat qiladi** (ehtimolliklar bilan). Shuningdek ertangi shamol tezligini
hisoblab, shamol generatori qancha quvvat berishini taxmin qiladi.

## Nima uchun?
Shamol generatorim (B&S loyihalari) uchun: qaysi kunlari shamol yetarli bo'lishini oldindan bilish.

## Qanday ishlaydi
1. `fetch_data.py`: Open-Meteo'dan 2005-yildan hozirgacha kunlik ma'lumot yuklaydi.
2. `common.py`: diagnoz qoidalari va belgilar (features) yasash.
3. `train.py`: Random Forest modelini o'qitadi, baseline bilan solishtiradi.
4. `predict.py`: bugungi diagnoz, ertangi ehtimolliklar va shamol quvvati.

## Ishga tushirish
```
pip install -r requirements.txt
python fetch_data.py
python train.py
python predict.py
```

## Diagnoz qoidalari
| Tur | Shart |
|---|---|
| snow | qor >= 0.1 sm |
| rain | yog'in >= 1 mm |
| windy | maks. shamol >= 8 m/s |
| cloudy | bulutlilik >= 40% |
| sunny | qolgan holatlar |

## Natijalar
(`python train.py` natijasini shu yerga yozing: baseline aniqligi, model aniqligi, shamol MAE)

![Confusion matrix](results/confusion_matrix.png)

## Kelajakdagi ishlar
- Haqiqiy shamol sensori (ESP32 + anemometr) ma'lumotini qo'shish
- Bir necha kun oldinga bashorat
- Boshqa shaharlar uchun sinash
