
import time
from ultralytics import YOLO
import cv2
import numpy as np
import unicodedata
import textwrap
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

vermelho = (140, 0, 0)

amarelo = (230, 230, 0)

amareloBGR = (0, 230, 230)

verdeBGR = (100, 220, 150)

verde = (100, 220, 150)

branco = (240, 240, 240)

cinza = (40,40,40)

# Endereço do último treinamento da IA

model = YOLO("runs/detect/treinamento_6/weights/best.pt")

# Ativa a câmera

cap = cv2.VideoCapture(0)

# Resolução da câmera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1728)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 964)

if not cap.isOpened():
    print("Erro: nao foi possivel abrir a câmera.")
    print("Verifique se sua câmera está conectada.")
    exit()
    
# Pasta para armazenar as capturas

pasta_capturas = "Capturas"

os.makedirs(pasta_capturas, exist_ok=True)

contador_capturas = 0

# Informações sobre as ferramentas

ferramentas = {
    "fita_isolante": {
        "nome": "Fita Isolante",
        "finalidade": [
            "Isolar fios desencapados e conexões elétricas."
        ],
        "atividades": [
            "Pequenos reparos e manutenções elétricas."
        ],
        "cuidados": [
            "Respeitar a indicação de tensão suportada pela fita.",
            "Aplicar em locais limpos e secos."
        ]
    },

    "alicate_universal": {
        "nome": "Alicate Universal",
        "finalidade": [
            "Segurar, puxar, dobrar chapas, cortar fios e arames."
        ],
        "atividades": [
            "Pequenos reparos domésticos."
        ],
        "cuidados": [
            "Não utilizar como martelo.",
            "Verificar o estado das mandíbulas e do cabo."
        ],
        "objetos_relacionados": [
        {
            "nome": "Fios Elétricos",
            "imagem": "ImagensRelacionamento/FioEletrico.jpg"
        }
        ]
    },

    "chave_philips": {
        
        "nome": "Chave Phillips",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe em cruz."
        ],
        "atividades": [
            "Montagem de móveis.",
            "Manutenção de equipamentos."
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel.",
            "Não utilizar em instalacoes eletricas energizadas ",
            "se a ferramenta não for apropriada para isso."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Parafusos com Cruz",
            "imagem": "ImagensRelacionamento/ParafusoCruz.jpg"
        }
        ]
    },

    "chave_de_fenda": {
        "nome": "Chave de fenda",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe reto."
        ],
        "atividades": [
            "Montagem de móveis.",
            "Manutenção de equipamentos eletrônicos."
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel.",
            "Não utilizar em instalacoes eletricas energizadas "
            "se a ferramenta nao for apropriada para isso."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Parafusos com Fenda",
            "imagem": "ImagensRelacionamento/ParafusoFenda.jpg"
        }
        ]
    },
    
    "chave_allen": {
        "nome": "Chave Allen",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe hexagonal."
        ],
        "atividades": [
            "Mecânica Automotiva."
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel."
        ]
    },
    
    "chave_combinada": {
        "nome": "Chave Combinada",
        "finalidade": [
            "Apertar e soltar porcas e parafusos."
        ],
        "atividades": [
            "Mecânica."
        ],
        "cuidados": [
            "Não usar como alavanca."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Parafusos e Porcas Hexagonais",
            "imagem": "ImagensRelacionamento/PorcasParafusos.jpg"
        }
        ]
    },
     
    "grifo": {
        "nome": "Grifo",
        "finalidade": [
            "Apertar, desapertar ou girar tubos, canos e conexões metálicas."
        ],
        "atividades": [
            "Serviços de Hidráulica."
        ],
        "cuidados": [
            "Lubrifique o ajuste quando necessário.",
            "Não use como martelo."
        ]
    },

    "martelo": {
        "nome": "Martelo",
        "finalidade": [
            "Aplicar impactos para fixar pregos.",
            "Também é possível remove-los, utilizando o outro lado da cabeça."
        ],
        "atividades": [
            "Carpintaria e marcenária.",
            "Montagem de móveis."
        ],
        "cuidados": [
            "É recomendável usar óculos.",
            "Verificar se a cabeça está firme.",
            "Inspecionar o cabo, com o tempo ele pode apresentar: ",
            "ressecamento, farpas e rachaduras."
            
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Pregos para Madeira",
            "imagem": "ImagensRelacionamento/Pregos.jpg"
        }
        ]
    },
    
     "estilete": {
            "nome": "Estilete",
            "finalidade": [
                "Serve para realizar cortes limpos, controlados e precisos em diversos tipos de materiais."
            ],
            "atividades": [
                "Comunicação Visual",
                "Papelaria"
            ],
            "cuidados": [
                "Recomendável usar óculos, a lâmina pode escapar",
                "Prestar atenção na posição do seus dedos."
            ]
        },

    "marreta": {
        "nome": "Marreta",
        "finalidade": [
            "Derrubar paredes, quebrar pedras, tijolos e blocos."
        ],
        "atividades": [
            "Serviços de Demolição",
            "Alvenaria",
            "Construção Civil."
        ],
        "cuidados": [
            "Recomendável usar óculos e a depender do máterial, proteção: ",
            "auricular, para barulhos intensos.",
            "Verificar se a cabeca esta bem fixada.",
            "Inspecionar o cabo para identificar farpas ou rachaduras."
        ]
    },

    "trena": {
        "nome": "Trena",
        "finalidade": [
            "Medir distâncias e dimensões de objetos."
        ],
        "atividades": [
            "Serviços domésticos",
            "Engenharia Civil"
        ],
        "cuidados": [
            "Evitar recolher a fita bruscamente.",
            "Não deixe a trena pendurada apenas pela ponta da fita",
            "usando o peso da caixa para sustenta-la"
        ]
    },

    "paquimetro": {
        "nome": "Paquímetro",
        "finalidade": [
            "É utilizado para medir objetos pequenos com "
            "grande precisão."
        ],
        "atividades": [
            "Trabalhos de mecânica, estética, metalurgia e engenharia."
        ],
        "cuidados": [
            "Evitar quedas e impactos.",
            "Limpar as superficies de medicao após o uso."
        ]
    },

    "desencapador_de_fios": {
        "nome": "Desencapador de Fios",
        "finalidade": [
            "Corta e puxa a capa externa do fio de maneira limpa",
            "A maioria dos modelos possui lâmina para cortar cabos"
        ],
        "atividades": [
            "Elétrica.",
            "Eletrônica."
        ],
        "cuidados": [
            "Cortar e decapar fios elétricos dentro da faixa de espessura suportada pela ferramenta."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Fios Elétricos",
            "imagem": "ImagensRelacionamento/FioEletrico.jpg"
        }
        ]
    },

    "prensa_manual": {
        "nome": "Prensa Manual",
        "finalidade": [
            "Encaixe de peças por meio de pressão manual."
        ],
        "atividades": [
            "Mecânica."
        ],
        "cuidados": [
            "Utilizar graxa para lubrificar.",
            "Respeitar a capacidade da ferramenta, não exceda a força máxima."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Caixa de Direção",
            "imagem": "ImagensRelacionamento/CaixaDirecao.jpg"
        }
        ]
    },

    "crimpador": {
        "nome": "Crimpador",
        "finalidade": [
            "Serve para fixar de forma permanente terminais, conectores ou plugues em cabos elétricos."
        ],
        "atividades": [
            "Elétrica."
        ],
        "cuidados": [
            "Nunca tente prensar um cabo ou terminal maior do que a capacidade indicada",
            " nas matrizes (castanhas) do alicate."
        ],
        "objetos_relacionados": [
        {
            "nome": "Terminais elétricos",
            "imagem": "ImagensRelacionamento/Ilhós.jpg"
        }
]
    },

    "bomba_de_ar": {
        "nome": "Bomba de Ar",
        "finalidade": [
            "Calibrar ou encher pneus, suspensões, bolas de futebol etc."
        ],
        "atividades": [
            "Mecânica.",
            "Industrial."
        ],
        "cuidados": [
            "Nunca use óleo e outros derivados do petróleo para a lubrificação use sempre graxa sintética.",
            "Periodicamente, desmonte e limpe o tubo interno para remover detritos que reduzem a pressão."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Pneus de Bicicleta",
            "imagem": "ImagensRelacionamento/ParPneu.jpg"
        }
        
        ]
    },
     
    "chave_inglesa": {
        "nome": "Chave Inglesa",
        "finalidade": [
            "Usada para apertar ou soltar porcas e parafusos de diferentes tamanhos."
        ],
        "atividades": [
            "Mecânica."
        ],
        "cuidados": [
            "Não serve como Grifo, por isso não deve ser usada em Tubos."
        ],
        "objetos_relacionados": 
            [
        {
            "nome": "Porcas e Parafusos Hexagonais",
            "imagem": "ImagensRelacionamento/PorcasParafusos.jpg"
        }
        
        ]
    },
}

# Normaliza os nomes para facilitar a busca

def normalizar(texto):
    
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere for caractere in texto
        if not unicodedata.combining(caractere)
    )
    
    return texto.lower().replace(" ", "_")

# Permite localizar paquimetro mesmo se houver acento

ferramentas_normalizadas = {
    normalizar(chave): valor
    for chave, valor in ferramentas.items()
}

# Configuracao da janela

janela = "YOLO - Reconhecimento de Ferramentas"

cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
cv2.resizeWindow(janela, 1364, 964)

# Fontes para o painel, incluindo acentos

# Tamanho das fontes

try:
    fonte_titulo = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 28
    )
    fonte_subtitulo = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 26
    )
    fonte_texto = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 24
    )
    
except OSError:
    
    fonte_titulo = ImageFont.load_default()
    fonte_subtitulo = ImageFont.load_default()
    fonte_texto = ImageFont.load_default()

ultima_classe = None
ultima_confianca = 0.0

contador_capturas = 0

mensagem_captura = ""
tempo_mensagem = 0

# Cria o painel superior
painel_superior = np.zeros((50, 864, 3), dtype=np.uint8)

painel_superior[:] = cinza

cv2.rectangle(
    painel_superior,
    (0, 0),
    (863, 49),
    amareloBGR,
    2
)

cv2.putText(
    painel_superior,
    "RECONHECIMENTO EM TEMPO REAL",
    (170, 34),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    amareloBGR,
    2,
    cv2.LINE_AA
)

def desenha_objetos_relacionados(painel, desenho, objetos, y):

    if not objetos:
        return y

    desenho.text(
        (20, y),
        "Objetos relacionados",
        font=fonte_subtitulo,
        fill=verde
    )

    y += 40

    largura_imagem = 180
    altura_imagem = 180
    largura_coluna = 230
    altura_linha = 200

    for i, objeto in enumerate(objetos):

        coluna = i % 2
        linha = i // 2

        x = 20 + coluna * largura_coluna
        y_atual = y + linha * altura_linha

        if y_atual + altura_linha > painel.height - 10:
            break

        caminho = objeto["imagem"]

        if not os.path.exists(caminho):
            print(f"Imagem não encontrada: {caminho}")
            continue

        try:
            with Image.open(caminho) as imagem:

                imagem = imagem.convert("RGB")
                imagem.thumbnail((largura_imagem, altura_imagem))

                x_imagem = x + (largura_imagem - imagem.width) // 2

                painel.paste(imagem, (x_imagem, y_atual))

            nome = objeto["nome"]

            partes = textwrap.wrap(nome, width=24)

            for j, parte in enumerate(partes):
                desenho.text(
                    (x, y_atual + altura_imagem + 5 + j * 20),
                    parte,
                    font=fonte_texto,
                    fill=branco
                )

        except Exception as erro:
            print(f"Erro ao carregar imagem: {erro}")

    return y + ((len(objetos) + 1) // 2) * altura_linha

   

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem da camera !.")
        break

    # Detecta as ferramentas
    results = model.predict(
        frame,
        conf = 0.60, # Nível de confiança mínimo para mostrar a ferramenta
        verbose = False
    )

    resultado = results[0]
    
    
    # Imagem da câmera com as detecções
    camera = resultado.plot()

    # Recorta a imagem para ficar quadrada, sem distorcer
    altura, largura = camera.shape[:2]
    lado = min(altura, largura)

    inicio_x = (largura - lado) // 2
    inicio_y = (altura - lado) // 2

    camera = camera[
        inicio_y:inicio_y + lado,
        inicio_x:inicio_x + lado
    ]

    # Redimensiona para o tamanho desejado
    camera = cv2.resize(camera, (864, 864))
    
    # Cria o painel inferior para feedback da captura
    painel_inferior = np.zeros((50, 864, 3), dtype=np.uint8)

    painel_inferior[:] = cinza

    cv2.rectangle(
        painel_inferior,
        (0, 0),
        (863, 49),
        amareloBGR,
        2
    )

    if time.time() < tempo_mensagem:

        cv2.putText(
            painel_inferior,
            mensagem_captura,
            (150, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            verdeBGR,
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            painel_inferior,
            "TECLA C : CAPTURAR IMAGEM DA CÂMERA",
            (180, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            amareloBGR,
            2,
            cv2.LINE_AA
        )
    
    # Cria o painel lateral
    
    painel = Image.new("RGB", (500, 964), cinza)
    desenho = ImageDraw.Draw(painel)

    desenho.text(
        (20, 18),
        "INFORMAÇÕES DA FERRAMENTA",
        font = fonte_titulo,
        fill = amarelo
    )

    desenho.line( #LINHA
        (20, 55, 475, 55),
        fill=amarelo,
        width=2
    )

    # Verifica se alguma ferramenta foi detectada
    
    if resultado.boxes is not None and len(resultado.boxes) > 0:

        melhor = int(resultado.boxes.conf.argmax())
        classe_id = int(resultado.boxes.cls[melhor])

        ultima_classe = model.names[classe_id]
        ultima_confianca = float(resultado.boxes.conf[melhor])

    # Exibe as informações da última ferramenta reconhecida
    
    if ultima_classe is not None:
        
        classe = ultima_classe
        confianca = ultima_confianca

        chave = normalizar(classe)
        info = ferramentas_normalizadas.get(chave)
        
        y = 80
          
        if info:
            
            # Exibe os quatro topicos
            
            topicos = [
                ("Nome", "nome"),
                ("Finalidade", "finalidade"),
                ("Atividades", "atividades"),
                ("Cuidados", "cuidados")
            ]

            for titulo, chave_topico in topicos:
                linhas = info.get(chave_topico, [])

                # Não mostra tópicos vázios
                
                if not linhas:
                    continue

                # Titulo do tópico
                
                desenho.text(
                    (20, y),
                    titulo,
                    font = fonte_subtitulo,
                    fill = amarelo
                )

                y += 32
                
                if chave_topico == "nome":
                    texto_nome = info.get("nome", "")
                                                
                    desenho.text(
                        (20, y),
                        texto_nome,
                        font = fonte_texto,
                        fill = branco
                    )
                                                
                    y += 36
                                                                
                    continue

                # Descrição com quebra automatica de linha
                
                for texto in linhas:
                    partes = textwrap.wrap(
                        texto,
                        width=43
                    )

                    for parte in partes:
                        if y > 565:
                            desenho.text(
                                (20, y - 5),
                                "...",
                                font = fonte_texto,
                                fill = branco
                            )
                            break

                        desenho.text(
                            (20, y),
                            parte,
                            font = fonte_texto,
                            fill = branco
                        )
                        y += 24

                    if y > 565:
                        break

                y += 12

                if y > 565:
                    break
                
            objetos = info.get("objetos_relacionados", [])

            y = desenha_objetos_relacionados(
                painel,
                desenho,
                objetos,
                y
            )

        else:
            desenho.text(
                (20, y),
                "A descrição ainda não foi cadastrada.",
                font = fonte_texto,
                fill = vermelho
            )

    else:
        desenho.text(
            (20, 135),
            "Aponte a câmera para uma ferramenta.",
            font = fonte_texto,
            fill = verde
        ),
        
        desenho.text(
            (20, 100),
            "Nenhuma ferramenta detectada.",
            font = fonte_texto,
            fill = vermelho
        )
        
    # Converte o painel para o formato de cor do OpenCV
    painel = cv2.cvtColor(
        np.array(painel),
        cv2.COLOR_RGB2BGR
    )

    camera_com_paineis = np.vstack((
    painel_superior,
    camera,
    painel_inferior
))

    tela = np.hstack((camera_com_paineis, painel))
    
    cv2.imshow(janela, tela)

    tecla = cv2.waitKey(1)

    if tecla != -1:
        tecla = tecla & 0xFF

    # Pressione C para capturar uma imagem
    
    if tecla == ord("c") or tecla == ord("C"):

        contador_capturas += 1

        nome_arquivo = f"captura_{contador_capturas:03d}.jpg"

        caminho = os.path.join(
        pasta_capturas,
        nome_arquivo
        )

        cv2.imwrite(caminho, frame)

        print(f"Imagem salva: {caminho}")
        
        mensagem_captura = "IMAGEM CAPTURADA COM SUCESSO !"
        tempo_mensagem = time.time() + 3
   
    # Pressione Q para encerrar
    if tecla == ord("q") or tecla == ord("Q") or tecla == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
