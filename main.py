from app.cli import AppCLI


## TODO  arapça sözsonu ünsüz ikilemesi
## his hissim zan zannetmek  vs
##TODO improved beam search?
## TODO ler ve le r ayrışımını nasıl öğreteceğiz?
## TODO  etimolojiyi bırak, mış, acak, ır eklerini conjugation almaya zorla. aynı zincirde bir tane.
## TODO maksızın anlaşılmıyor, ın geri mi gelsin yoksa maksızın mı olsun
## ır mış acak isimleşmişler sözlükte var diyip ardına kişiçekim almayı zorunlu mu kılsam?
## tek başına her sözcük o kişisi ile cümle . O araba. 0 suffix yerine 1 suffix daha iyi.
#  e o zaman her sözcük null suffix alacak. O zaman cümlede bir tane conjugation olmalı diye bir şey mi yapsam?
if __name__ == "__main__":
    
    cli = AppCLI()
    cli.run()


