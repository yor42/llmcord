import base64
import json

import png
import zlib

import png
import base64

class CharacterCardManager:

    def extract_text_chunk(png_file):
        """
        Extract and decode a specific base64-encoded tEXt chunk from a PNG file

        Args:
            png_file (str): Path to the PNG file
            chunk_name (list): Name of the tEXt chunk to extract

        Returns:
            dict: Json of Decoded text chunk data, or None if not found
        """

        chunk_name = ['ccv3', 'chara']


        try:
            with open(png_file, 'rb') as f:
                # PNG signature check
                signature = f.read(8)
                if signature != b'\x89PNG\r\n\x1a\n':
                    raise ValueError("Not a valid PNG file")

                while True:
                    # Read chunk length
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break

                    length = int.from_bytes(length_bytes, 'big')

                    # Read chunk type
                    chunk_type = f.read(4)

                    # Check if it's a tEXt chunk
                    if chunk_type == b'tEXt':
                        chunk_data = f.read(length)

                        # Split keyword and text
                        null_index = chunk_data.index(b'\x00')
                        keyword = chunk_data[:null_index].decode('utf-8')

                        if keyword in chunk_name:
                            base64_data = chunk_data[null_index + 1:]
                            # Decode base64 and then UTF-8
                            decoded_text = base64.b64decode(base64_data).decode('utf-8')
                            decoded_json = json.loads(decoded_text)

                            if keyword != 'chara' and decoded_json["spec"] != "chara_card_v3":
                                print(f"WARN: This card is not following 3.0 Standard! We will still try to process it, but Here be dragons!")
                            else:
                                version = "2.0"
                                if "spec_version" in decoded_json:
                                    version = decoded_json["spec_version"]
                                print(f"Found Character card with spec version {version}!" )
                            return decoded_json["data"]


                    # Skip chunk data and CRC
                    f.seek(length + 4, 1)

            return None

        except Exception as e:
            print(f"Error reading PNG: {e}")
            return None



    encoding="utf-8"

    if __name__ == "__main__":
        file = input("input image name:")
        fullpath = "./charcard/"+file+".png"
        data = extract_text_chunk(fullpath)
        print(data)

        with open("./characters/"+file+".json", "w", encoding=encoding) as jsonfile:
            json.dump(data, jsonfile)

        if "character_book" in data:
            lorebooklist = data["character_book"]["entries"]
            if len(lorebooklist)>0:
                with open("./lorebooks/"+file+".json", "w", encoding=encoding) as jsonfile:
                    json.dump(lorebooklist, jsonfile)
        input("Press Enter to continue...")