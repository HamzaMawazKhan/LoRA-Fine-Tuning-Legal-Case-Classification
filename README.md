# Legal Document Processing & Mistral Fine-Tuning

A comprehensive toolkit for processing legal documents from PDFs, extracting case information, and fine-tuning Mistral-7B models for legal applications using LoRA (Low-Rank Adaptation).

## 🚀 Features

- **PDF Text Extraction**: Extract and clean text from legal PDF documents
- **Case Classification**: Automatically categorize legal cases based on extracted information
- **Dataset Generation**: Create labeled and unlabeled datasets from legal documents
- **LoRA Fine-Tuning**: Fine-tune Mistral-7B-Instruct on legal data using parameter-efficient training
- **Inference Pipeline**: Generate legal responses using the fine-tuned model

## 📋 Requirements

### System Requirements
- macOS with Apple Silicon (M1/M2/M3) or CUDA-compatible GPU
- Python 3.8+
- At least 16GB RAM (32GB recommended for training)

### Dependencies

```bash
pip install transformers datasets peft accelerate
pip install PyMuPDF pandas torch
```

For Apple Silicon:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📁 Project Structure

```
legal-ai-toolkit/
├── pdf_processor.py          # PDF text extraction and cleaning
├── mistral_fine_tuner.py     # LoRA fine-tuning script
├── inference.py              # Model inference pipeline
├── data/
│   ├── nyscef/              # PDF documents directory
│   ├── cases.csv            # Case labels and metadata
│   ├── labeled_data.json    # Processed labeled dataset
│   └── unlabeled_data.json  # Processed unlabeled dataset
├── models/
│   └── legal-mistral-lora/  # Fine-tuned model output
└── README.md
```

## ⚡ Quick Start

### 1. Dataset Preparation

First, process your legal PDFs and create training datasets:

```python
from pdf_processor import build_dataset

# Process PDFs and create labeled/unlabeled datasets
build_dataset(
    pdf_dir="data/nyscef/",
    csv_label_path="data/cases.csv",
    output_labeled="data/labeled_data.json",
    output_unlabeled="data/unlabeled_data.json",
    skip_unlabeled=False
)
```

### 2. Model Fine-Tuning

Fine-tune Mistral-7B-Instruct on your legal dataset:

```python
# Run the fine-tuning script
python mistral_fine_tuner.py
```

**Configuration Options:**
- `model_id`: Base model identifier (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `data_path`: Path to your training dataset JSON file
- `output_dir`: Directory to save the fine-tuned model
- `max_length`: Maximum sequence length (default: 1024)
- `batch_size`: Training batch size (default: 1 for Apple Silicon)

### 3. Inference

Use the fine-tuned model for legal question answering:

```python
from inference import generate

# Generate legal advice or analysis
response = generate("What are the legal consequences of fraud?")
print(response)
```

## 📊 Dataset Format

The system expects your training data in JSON format with prompt-response pairs:

```json
[
  {
    "prompt": "What are the elements of a breach of contract claim?",
    "response": "A breach of contract claim typically requires four elements: (1) existence of a valid contract, (2) performance by the plaintiff, (3) breach by the defendant, and (4) damages resulting from the breach."
  }
]
```

For PDF processing, ensure your CSV labels file contains:
- `case_number`: Unique case identifier
- `case_type`: Legal category/classification

## ⚙️ Training Configuration

### LoRA Parameters
- **r**: 16 (rank of adaptation)
- **lora_alpha**: 32 (scaling parameter)
- **lora_dropout**: 0.1 (dropout rate)
- **target_modules**: All linear layers

### Training Parameters
- **epochs**: 3
- **learning_rate**: Auto-determined by Trainer
- **batch_size**: 1 (with gradient accumulation of 4 steps)
- **precision**: bfloat16 (for Apple Silicon compatibility)

## 🧠 Memory Optimization

For Apple Silicon devices with limited memory:

1. **Gradient Accumulation**: Increase `gradient_accumulation_steps` to simulate larger batch sizes
2. **Sequence Length**: Reduce `max_length` if facing memory issues
3. **LoRA Rank**: Lower `r` parameter to reduce trainable parameters
4. **Precision**: Use `bf16=True` for better memory efficiency

## 🔧 Model Loading

The fine-tuned model can be loaded in two ways:

### Method 1: Direct Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./legal-mistral-lora")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "./legal-mistral-lora")
```

### Method 2: Config-based Loading
```python
from peft import PeftConfig

config = PeftConfig.from_pretrained("./legal-mistral-lora")
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "./legal-mistral-lora")
```

## 📄 PDF Processing Features

The PDF processor includes:
- **Text Extraction**: Uses PyMuPDF for robust PDF text extraction
- **Case ID Detection**: Automatically extracts INDEX NO. from legal documents
- **Text Cleaning**: Removes formatting artifacts and normalizes text
- **Error Handling**: Gracefully handles corrupted or unreadable PDFs

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or max sequence length
2. **PDF Extraction Failures**: Ensure PDFs are text-based, not scanned images
3. **Model Loading Issues**: Verify all dependencies are installed correctly
4. **Training Stalls**: Check if MPS is available on Apple Silicon

### Performance Tips

- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable gradient checkpointing for memory efficiency
- Use mixed precision training with `bf16=True`

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 💬 Support

For questions and support, please open an issue on GitHub or contact [hamzamawazk@gmail.com].

---

⚖️ **Disclaimer**: This tool is for educational and research purposes only. Always consult with qualified legal professionals for actual legal advice.
