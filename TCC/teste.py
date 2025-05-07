import os
from PIL import Image, ImageDraw, ImageFont
import math # Importar o módulo math para usar ceil

def combine_images(root_dir, output_path):
    """
    Combina imagens de subpastas de um determinado diretório em uma única imagem,
    com o nome de cada subpasta como um rótulo acima de suas imagens combinadas,
    organizando-as em um grid.

    Args:
        root_dir (str): O diretório raiz contendo subpastas com imagens.
        output_path (str): O caminho para salvar a imagem combinada.
    """

    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    images_data = []  # Lista para armazenar o nome da subpasta e os caminhos das imagens

    for subfolder in subfolders:
        image_paths = sorted([
            os.path.join(subfolder, f.name)
            for f in os.scandir(subfolder)
            if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ])  # Lida com vários formatos de imagem

        if image_paths:  # Adiciona apenas se houver imagens na subpasta
          images_data.append((os.path.basename(subfolder), image_paths))

    if not images_data:
        print("Erro: Nenhuma imagem encontrada no diretório especificado ou em suas subpastas.")
        return

    # Configuração para o layout final da imagem
    images_per_row = 2  # <--- ALTERADO: Número de blocos de subpasta por linha
    padding = 20 # Aumentei um pouco o padding para melhor visualização
    font_size = 25 # Aumentei um pouco o tamanho da fonte
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use uma fonte comum
    except IOError:
        font = ImageFont.load_default()
        print("Aviso: Fonte 'arial.ttf' não encontrada, usando fonte padrão.")


    # Calcular dimensões máximas de uma única imagem em todas as subpastas
    max_img_width = 0
    max_img_height = 0
    for _, img_paths in images_data:
        for img_path in img_paths:
            try:
                with Image.open(img_path) as img:
                    max_img_width = max(max_img_width, img.width)
                    max_img_height = max(max_img_height, img.height)
            except FileNotFoundError:
                print(f"Aviso: Imagem não encontrada: {img_path}")
            except Exception as e:
                print(f"Aviso: Não foi possível processar a imagem {img_path}. Erro: {e}")


    # Calcular a altura de cada bloco de subpasta (imagens empilhadas + rótulo)
    # A altura de um bloco é a soma das alturas das imagens + padding entre elas + altura do texto + padding
    block_heights = []
    for subfolder_name, image_paths in images_data:
        current_block_height = font_size + padding # Altura inicial para o rótulo
        for img_path in image_paths:
             try:
                 with Image.open(img_path) as img:
                    current_block_height += max_img_height + padding # Usamos a altura máxima para manter o alinhamento
             except (FileNotFoundError, Exception):
                 pass # Já tratamos erros na primeira passagem

        block_heights.append(current_block_height)

    max_block_height = max(block_heights) if block_heights else 0


    # Calcular a largura de cada bloco (baseado na largura máxima da imagem)
    block_width = max_img_width + 2 * padding # Largura da imagem + padding nas laterais do bloco

    # Calcular o número de linhas necessárias
    num_subfolders = len(images_data)
    num_rows = math.ceil(num_subfolders / images_per_row)

    # Calcular as dimensões totais da imagem final
    total_width = images_per_row * block_width + (images_per_row + 1) * padding
    total_height = num_rows * max_block_height + (num_rows + 1) * padding # Altura total considerando o padding entre as linhas

    # Criar a imagem final
    final_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(final_image)

    # Posicionar os blocos na imagem final
    x_offset = padding
    y_offset = padding
    for i, (subfolder_name, image_paths) in enumerate(images_data):
        # Desenhar o nome da subpasta
        text_width = draw.textlength(subfolder_name, font=font)
        text_x = x_offset + (block_width - text_width) // 2
        draw.text((text_x, y_offset), subfolder_name, fill="black", font=font)

        current_img_y_offset = y_offset + font_size + padding # Posição Y para a primeira imagem no bloco

        # Colar imagens do bloco atual
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                # Centralizar a imagem dentro do espaço alocado para ela
                img_x = x_offset + (block_width - img.width) // 2 # Centraliza horizontalmente no bloco
                final_image.paste(img, (img_x, current_img_y_offset))
                current_img_y_offset += max_img_height + padding # Mover para a próxima posição de imagem no bloco
            except FileNotFoundError:
                print(f"Aviso: Imagem não encontrada: {img_path}")
            except Exception as e:
                print(f"Aviso: Não foi possível processar a imagem {img_path}. Erro: {e}")

        # Atualizar offset para o próximo bloco
        if (i + 1) % images_per_row == 0:
            # Ir para a próxima linha
            x_offset = padding
            y_offset += max_block_height + padding
        else:
            # Ir para a próxima coluna na mesma linha
            x_offset += block_width + padding

    final_image.save(output_path)
    print(f"Imagem combinada salva em {output_path}")

if __name__ == "__main__":
    root_dir = "C:/Users/Hmmm1/OneDrive/Documentos/TCC/resuts/fraud_dataset/tunning/nn/1000/"
    output_path = "C:/Users/Hmmm1/OneDrive/Documentos/TCC/resuts/fraud_dataset/combined_images.png"
    combine_images(root_dir, output_path)