/opt/homebrew/bin/python3.11 -m venv quantum_env_311
rm -rf quantum_env_311
source quantum_env_311/bin/activate
pip install -r requirements.txt
pip install --upgrade pip

```mermaid
stateDiagram-v2


    [*] --> Start

    %% Basic word processing
    Start --> Word: letter
    Word --> Word: letter
    Word --> WordEnd: space
    WordEnd --> Start: space
    
    %% Punctuation handling
    Start --> Punctuation: punct
    Word --> Punctuation: punct
    Punctuation --> Word: letter
    Punctuation --> WordEnd: space
    
    %% Multi-word expression path
    Start --> MWEFirst: capital
    MWEFirst --> MWEFirst: letter
    MWEFirst --> MWESpace: space
    MWESpace --> MWENext: capital
    MWENext --> MWENext: letter
    MWENext --> MWEEnd: space/punct
    
    %% Contractions and special cases
    Word --> Apostrophe: apostrophe
    Apostrophe --> Contraction: letter/n
    Contraction --> Contraction: letter
    Contraction --> WordEnd: space/punct
    
    %% Final states
    WordEnd --> [*]
    MWEEnd --> [*]
    
    %% Styling
    class Word wordState
    class Punctuation punctState
    class MWEFirst mweState
    class MWENext mweState
    class Contraction specialState
    
    %% Example annotations
    note right of Word
        Basic words:
        "cat", "dog", "the"
    end note
    
    note right of MWEFirst
        Multi-word expressions:
        "New York"
        "San Francisco"
        "rock 'n' roll"
    end note
    
    note right of Contraction
        Contractions:
        "don't"
        "it's"
        "rock 'n' roll"
    end note
    
    note left of Punctuation
        Punctuation:
        ",", ".", "!", "?"
        "(", ")", "[", "]"
    end note
```
install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh


uv pip compile pyproject.toml --generate-hashes -o requirements.txt

uv pip sync requirements.txt

uv pip compile requirements.txt --refresh
uv sync # generated uv.lock



chmod +x entrypoint.sh