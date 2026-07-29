import json
import logging
from pathlib import Path
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature

logger = logging.getLogger(__name__)

class CairoABIParser:
    def __init__(self):
        # Маппинг базовых типов Cairo на их размер в байтах. 
        # felt252 и ContractAddress занимают 31-32 байта (1 поле).
        self.type_size_map = {
            "u8": 1, "u16": 2, "u32": 4, "u64": 8, 
            "u128": 16, "u256": 32, "felt252": 32, 
            "ContractAddress": 32, "bool": 1
        }

    def parse_file(self, filepath: str | Path) -> list[ContractFeature]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                abi = json.load(f)
            return self.parse_abi(abi)
        except FileNotFoundError:
            logger.error(f"ABI file not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in ABI file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse Cairo ABI file {filepath}: {e}", exc_info=True)
            raise

    def parse_abi(self, abi: list[dict]) -> list[ContractFeature]:
        features = []
        try:
            # В современных версиях Cairo переменные состояния лежат внутри структуры, 
            # имя которой заканчивается на "::Storage"
            storage_struct = next(
                (item for item in abi if item.get("type") == "struct" and "Storage" in item.get("name", "")), 
                None
            )
            
            if not storage_struct:
                logger.warning("No Storage struct found in ABI. Returning empty feature list.")
                return features

            for member in storage_struct.get("members", []):
                name = member.get("name", "unknown")
                cairo_type = member.get("type", "")
                
                # Поиск размера по паттерну (fallbacks до 32 байт, если это сложный тип)
                size = 32 
                for t_name, t_size in self.type_size_map.items():
                    if t_name in cairo_type:
                        size = t_size
                        break
                        
                # Переменные в Storage всегда потенциально изменяемы
                features.append(ContractFeature(
                    name=name,
                    data_type=cairo_type,
                    size_bytes=size,
                    is_mutable=True
                ))
                
            logger.info(f"Successfully extracted {len(features)} storage variables from Cairo ABI.")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from ABI structure: {e}", exc_info=True)
            raise