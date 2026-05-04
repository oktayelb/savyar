# Türkçe için kök-ek ayrıştırıcısı (SAV-YAR)

Kural tabanlı olası ayrışımlar arasından ML doğru olanı sıralamayı öğreniyor.
Öncelikle kural tabanlı bir ayrıştırıcı cümledeki sözcüklerin olası bütün kök-ek ayrışımlarını veriyor.
Bu aşama Türkçe eklerinin koda gömülme aşamasıdır. Cümle bazında kurallar değil sözcük bazında ek kuralları işlenir.

Sonrasında bu olası kök ek ayrışımları, kök bilgisi gözardı edilerek bir ek listesine dönüştürülür ve ML'e gönderilir. ML sözcüklerin kök bilgileri hakkında hiçbir bilgiye sahip değildir. Cümleleri yalnızca ekler olarak görür.

Eğitim seti içindeki her cümle için doğru ayışım bilgisi halihazırda bulunuyor. 
Eğitim sırasında örnek cümle önce olası bütün ayrışımlarına ayırılır,  
Bu olası bütün ayrışımlar arasından doğru ayrışım dışındaki her örnek yanlış olarak algılanır ve model cümleyi yalnış sıralamalar arasında nasıl sıralayacağını öğrenir.
15.000 örnek cümle ile cümle başında 10 yanlış ayrışımı eğitim seti olarak görür.

Şu anda cümle sıralama doğruluğu (Rank Accuracy) = 86

Şu anda tekil ek doğruluğu (Suffix Accuracy) = 95

---


### Epoch 1/10
- **Train**
  - Loss: 3.5851
  - Rank Loss: 1.9889
  - Rank Accuracy: 0.7098
  - Margin: 0.1843
- **Validation**
  - Rank Loss: 0.6835 
  - Rank Accuracy: 0.7697
  - SuffAcc: 0.9068
  - Precision: 0.8935
  - Recall: 0.8885
  - F1: 0.8910
  - Margin: 0.1915

---

### Epoch 2/10
- **Train**
  - Loss: 1.4362
  - Rank Loss: 0.7383
  - Rank Accuracy: 0.8066
  - Margin: 0.2591
- **Validation**
  - Rank Loss: 0.5898 
  - Rank Accuracy: 0.7767
  - SuffAcc: 0.9105
  - Precision: 0.9083
  - Recall: 0.9047
  - F1: 0.9065
  - Margin: 0.2613

---

### Epoch 3/10
- **Train**
  - Loss: 1.0829
  - Rank Loss: 0.6295
  - Rank Accuracy: 0.8239
  - Margin: 0.2737
- **Validation**
  - Rank Loss: 0.5301 
  - Rank Accuracy: 0.8231
  - SuffAcc: 0.9327
  - Precision: 0.9280
  - Recall: 0.9271
  - F1: 0.9275
  - Margin: 0.2044

---

### Epoch 4/10
- **Train**
  - Loss: 0.9403
  - Rank Loss: 0.5729
  - Rank Accuracy: 0.8358
  - Margin: 0.2899
- **Validation**
  - Rank Loss: 0.5100 
  - Rank Accuracy: 0.8242
  - SuffAcc: 0.9355
  - Precision: 0.9364
  - Recall: 0.9341
  - F1: 0.9352
  - Margin: 0.2488

---

### Epoch 5/10
- **Train**
  - Loss: 0.8487
  - Rank Loss: 0.5263
  - Rank Accuracy: 0.8458
  - Margin: 0.2966
- **Validation**
  - Rank Loss: 0.4754 
  - Rank Accuracy: 0.8373
  - SuffAcc: 0.9416
  - Precision: 0.9477
  - Recall: 0.9452
  - F1: 0.9465
  - Margin: 0.2832

---

### Epoch 6/10
- **Train**
  - Loss: 0.7919
  - Rank Loss: 0.4910
  - Rank Accuracy: 0.8537
  - Margin: 0.3108
- **Validation**
  - Rank Loss: 0.4612 
  - Rank Accuracy: 0.8357
  - SuffAcc: 0.9408
  - Precision: 0.9454
  - Recall: 0.9434
  - F1: 0.9444
  - Margin: 0.2946

---

### Epoch 7/10
- **Train**
  - Loss: 0.7482
  - Rank Loss: 0.4661
  - Rank Accuracy: 0.8616
  - Margin: 0.3281
- **Validation**
  - Rank Loss: 0.4700
  - Rank Accuracy: 0.8384
  - SuffAcc: 0.9441
  - Precision: 0.9497
  - Recall: 0.9477
  - F1: 0.9487
  - Margin: 0.3566

---

### Epoch 8/10
- **Train**
  - Loss: 0.7080
  - Rank Loss: 0.4381
  - Rank Accuracy: 0.8655
  - Margin: 0.3412
- **Validation**
  - Rank Loss: 0.4301 
  - Rank Accuracy: 0.8455
  - SuffAcc: 0.9486
  - Precision: 0.9516
  - Recall: 0.9499
  - F1: 0.9507
  - Margin: 0.3257

---

### Epoch 9/10
- **Train**
  - Loss: 0.6792
  - Rank Loss: 0.4173
  - Rank Accuracy: 0.8736
  - Margin: 0.3525
- **Validation**
  - Rank Loss: 0.4217 
  - Rank Accuracy: 0.8466
  - SuffAcc: 0.9466
  - Precision: 0.9495
  - Recall: 0.9475
  - F1: 0.9485
  - Margin: 0.3401

---

### Epoch 10/10
- **Train**
  - Loss: 0.6585
  - Rank Loss: 0.4036
  - Rank Accuracy: 0.8761
  - Margin: 0.3596
- **Validation**
  - Rank Loss: 0.4260
  - Rank Accuracy: 0.8472
  - SuffAcc: 0.9475
  - Precision: 0.9499
  - Recall: 0.9485
  - F1: 0.9492
  - Margin: 0.3469