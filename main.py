
from ultralytics import YOLO
import cv2
import numpy as np
import unicodedata
import textwrap
from PIL import Image, ImageDraw, ImageFont

# Endereço do último treinamento da IA

model = YOLO("runs/detect/treinamento_5/weights/best.pt")

# Ativa a câmera

cap = cv2.VideoCapture(0)

# Resolução da câmera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1728)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 864)

if not cap.isOpened():
    print("Erro: nao foi possivel abrir a câmera.")
    print("Verifique se sua câmera está conectada.")
    exit()

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
            "Respeitar a indicação de tensão suportada pela fita."
            "Aplicar em locais limpos e secos."
        ]
    },

    "alicate_universal": {
        "nome": "Alicate Universal",
        "finalidade": [
            "Segurar, puxar, dobrar chapas, cortar fios e arames."
        ],
        "atividades": [
            "Pequenos reparos domésticos.",
        ],
        "cuidados": [
            "Não utilizar como martelo.",
            "Verificar o estado das mandíbulas e do cabo."
        ]
    },

    "chave_philips": {
        "nome": "Chave Phillips",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe em cruz."
        ],
        "atividades": [
            "Montagem de moveis.",
            "Manutenção de equipamentos.",
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel.",
            "Não utilizar em instalacoes eletricas energizadas ",
            "se a ferramenta não for apropriada para isso."
        ]
    },

    "chave_de_fenda": {
        "nome": "Chave de fenda",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe reto."
        ],
        "atividades": [
            "Montagem de móveis.",
            "Manutenção de equipamentos eletrônicos.",
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel.",
            "Não utilizar em instalacoes eletricas energizadas "
            "se a ferramenta nao for apropriada para isso."
        ]
    },
    
    "chave_de_allen": {
        "nome": "Chave Allen",
        "finalidade": [
            "Apertar e soltar parafusos com encaixe hexagonal."
        ],
        "atividades": [
            "Mecânica Automotiva."
        ],
        "cuidados": [
            "Não utilizar como alavanca ou cinzel.",
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
            "Não usar como alavanca",
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
            "Lubrifique o ajuste quando necessário"
            "Não use como martelo.",
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
            "Verificar se a cabeca está firme.",
            "Inspecionar o cabo, com o tempo ele pode apresentar: ",
            "ressecamento, farpas e rachaduras."
            
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
                "Prestar atenção na posição dos dedos"
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
            "Construção Cívil.",
        ],
        "cuidados": [
            "Recomendável usar óculos e a depender do máterial, proteção: ",
            "auricular, para barulhos intensos."
            "Verificar se a cabeca esta bem fixada.",
            "Inspecionar o cabo para identificar farpas ou rachaduras.",
        ]
    },

    "trena": {
        "nome": "Trena",
        "finalidade": [
            "Medir distâncias e dimensoes de objetos."
        ],
        "atividades": [
            "Serviços domésticos",
            "Engenharia Cívil"
        ],
        "cuidados": [
            "Evitar recolher a fita bruscamente."
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
        ]
    },

    "prensa_manual": {
        "nome": "Prensa Manual",
        "finalidade": [
            "Encaixe de peças por meio de pressão manual"
        ],
        "atividades": [
            "Mecânica."
        ],
        "cuidados": [
            "Utilizar graxa para lubrificar."
            "Respeitar a capacidade da ferramenta, não exceda a força máxima."
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
        ]
    }
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
cv2.resizeWindow(janela, 1364, 864)

# Fontes para o painel, incluindo acentos

try:
    fonte_titulo = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 24
    )
    fonte_subtitulo = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 19
    )
    fonte_texto = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf", 16
    )
    
except OSError:
    
    fonte_titulo = ImageFont.load_default()
    fonte_subtitulo = ImageFont.load_default()
    fonte_texto = ImageFont.load_default()

ultima_classe = None
ultima_confianca = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem da camera.")
        break

    # Detecta as ferramentas
    results = model.predict(
        frame,
        conf = 0.79, # Nível de confiança mínimo para mostrar a ferramenta
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

    # Cria o painel lateral
    
    #painel = Image.new("RGB", (500, 864), (35, 35, 35))
    painel = Image.new("RGB", (500, 864), (10, 10, 10))
    desenho = ImageDraw.Draw(painel)

    verde = (100, 220, 150)
    branco = (240, 240, 240)
    cinza = (190, 190, 190)

    desenho.text(
        (20, 18),
        "INFORMAÇÕES DA FERRAMENTA",
        font=fonte_titulo,
        fill=verde
    )

    desenho.line(
        (20, 55, 475, 55),
        fill=(80, 80, 80),
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
                    font=fonte_subtitulo,
                    fill=verde
                )

                y += 28
                
                if chave_topico == "nome":
                    texto_nome = info.get("nome", "")
                                                
                    desenho.text(
                        (20, y),
                        texto_nome,
                        font=fonte_texto,
                        fill=branco
                    )
                                                
                    y += 34
                                                                
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
                                font=fonte_texto,
                                fill=cinza
                            )
                            break

                        desenho.text(
                            (20, y),
                            parte,
                            font=fonte_texto,
                            fill=branco
                        )
                        y += 22

                    if y > 565:
                        break

                y += 12

                if y > 565:
                    break

        else:
            desenho.text(
                (20, y),
                "Descrição ainda não cadastrada.",
                font=fonte_texto,
                fill=cinza
            )

    else:
        desenho.text(
            (20, 100),
            "Nenhuma ferramenta detectada.",
            font=fonte_texto,
            fill=cinza
        )

        desenho.text(
            (20, 135),
            "Aponte a câmera para uma ferramenta.",
            font=fonte_texto,
            fill=cinza
        )

    # Converte o painel para o formato de cor do OpenCV
    painel = cv2.cvtColor(
        np.array(painel),
        cv2.COLOR_RGB2BGR
    )

    # Junta a camera e o painel lateral
    tela = np.hstack((camera, painel))

    cv2.imshow(janela, tela)

    # Pressione Q para encerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
