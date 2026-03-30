# Vysvetlenie

Canny detektor hrán je vylepšený Sobelov operátor na detekciu hrán. Narozdiel od sobela, ktorý hľadá gradient canny hľadá ostré hrany. 

# Postup:

• Gaussovho filtrovania,
• výpočtu gradientu - sobelov operátor,
• výpočtu veľkosti a smeru gradientu,

• non-maximum suppression,
• dvojitého prahovania,
• hysterézie.


# Porovnanie s OpenCV a ukážka výsledkov:

čas 1: 640x480
 - canny 8 ms
 - vlastny 26.644 s

čas 2: 1200x903
 - canny 17 ms
 - vlastny 91,689 s

čas 3: 1600x1126
 - canny 15 ms
 - vlastny 140,94 s

# Zhodnotenie výhod a nevýhod

??