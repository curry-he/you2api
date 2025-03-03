package handler

import (
    "bufio"
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
    "path/filepath"
    "regexp"
    "strings"
    "time"

    "github.com/PuerkitoBio/goquery"
    "github.com/google/uuid"
)

// TokenCount 定义了 token 计数的结构
type TokenCount struct {
    PromptTokens     int `json:"prompt_tokens"`
    CompletionTokens int `json:"completion_tokens"`
    TotalTokens      int `json:"total_tokens"`
}

const (
    MaxContextTokens = 2000 // 最大上下文 token 数
)

// YouChatResponse 定义了从 You.com API 接收的单个 token 的结构。
type YouChatResponse struct {
    YouChatToken string `json:"youChatToken"`
}

// OpenAIStreamResponse 定义了 OpenAI API 流式响应的结构。
type OpenAIStreamResponse struct {
    ID      string   `json:"id"`
    Object  string   `json:"object"`
    Created int64    `json:"created"`
    Model   string   `json:"model"`
    Choices []Choice `json:"choices"`
}

// Choice 定义了 OpenAI 流式响应中 choices 数组的单个元素的结构。
type Choice struct {
    Delta        Delta  `json:"delta"`
    Index        int    `json:"index"`
    FinishReason string `json:"finish_reason"`
}

// Delta 定义了流式响应中表示增量内容的结构。
type Delta struct {
    Content string `json:"content"`
}

// OpenAIRequest 定义了 OpenAI API 请求体的结构。
type OpenAIRequest struct {
    Messages []Message `json:"messages"`
    Stream   bool      `json:"stream"`
    Model    string    `json:"model"`
}

// Message 定义了 OpenAI 聊天消息的结构。
type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

// OpenAIResponse 定义了 OpenAI API 非流式响应的结构。
type OpenAIResponse struct {
    ID      string         `json:"id"`
    Object  string         `json:"object"`
    Created int64          `json:"created"`
    Model   string         `json:"model"`
    Choices []OpenAIChoice `json:"choices"`
}

// OpenAIChoice 定义了 OpenAI 非流式响应中 choices 数组的单个元素的结构。
type OpenAIChoice struct {
    Message      Message `json:"message"`
    Index        int     `json:"index"`
    FinishReason string  `json:"finish_reason"`
}

// ModelResponse 定义了 /v1/models 响应的结构。
type ModelResponse struct {
    Object string        `json:"object"`
    Data   []ModelDetail `json:"data"`
}

// ModelDetail 定义了模型列表中单个模型的详细信息。
type ModelDetail struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Created int64  `json:"created"`
    OwnedBy string `json:"owned_by"`
}

// modelMap 存储 OpenAI 模型名称到 You.com 模型名称的映射。 (OpenAI -> You.com)
var modelMap = map[string]string{
    "deepseek-reasoner":       "deepseek_r1",
    "deepseek-chat":           "deepseek_v3",
    "o3-mini-high":            "openai_o3_mini_high",
    "o3-mini-medium":          "openai_o3_mini_medium",
    "o1":                      "openai_o1",
    "o1-mini":                 "openai_o1_mini",
    "o1-preview":              "openai_o1_preview",
    "gpt-4o":                  "gpt_4o",
    "gpt-4o-mini":             "gpt_4o_mini",
    "gpt-4-turbo":             "gpt_4_turbo",
    "gpt-3.5-turbo":           "gpt_3.5",
    "claude-3-opus":           "claude_3_opus",
    "claude-3-sonnet":         "claude_3_sonnet",
    "claude-3.5-sonnet":       "claude_3_5_sonnet",
    "claude-3.5-haiku":        "claude_3_5_haiku",
    "gemini-1.5-pro":          "gemini_1_5_pro",
    "gemini-1.5-flash":        "gemini_1_5_flash",
    "llama-3.2-90b":           "llama3_2_90b",
    "llama-3.1-405b":          "llama3_1_405b",
    "mistral-large-2":         "mistral_large_2",
    "qwen-2.5-72b":            "qwen2p5_72b",
    "qwen-2.5-coder-32b":      "qwen2p5_coder_32b",
    "command-r-plus":          "command_r_plus",
    "claude-3-7-sonnet":       "claude_3_7_sonnet", // 添加显式映射
    "claude-3-7-sonnet-think": "claude_3_7_sonnet_thinking",
    "grok-2":                  "grok_2",          // 添加显式映射
    "gemini-2-flash":          "gemini_2_flash",      // 添加显式映射
}

// reverseModelMap 存储 You.com 模型名称到 OpenAI 模型名称的映射。 (You.com -> OpenAI)
var reverseModelMap map[string]string

// Dynamic model list and map
var dynamicModelMap = make(map[string]string)
var dynamicModels []ModelDetail
var useDynamicModels = false

// 定义 JSON 结构来解析 __NEXT_DATA__
type NextData struct {
    Props struct {
        PageProps struct {
            AiModels []struct { // 将 ChatModeModels 修改为 AiModels，并匹配 JSON 结构中的 aiModels
                ID                      string `json:"id"`
                Name                    string `json:"name"`
                Company                 string `json:"company"`
                IsProOnly               bool   `json:"isProOnly"`
                ContextLimit            int    `json:"contextLimit"`
                IsUncensoredModel       bool   `json:"isUncensoredModel"`
                IsAllowedForUserChatModes bool   `json:"isAllowedForUserChatModes"`
                ImageURL                string `json:"imageUrl"`
                Tagline                 string `json:"tagline"`
            } `json:"aiModels"` // 确保 JSON tag 与 JSON 结构一致
        } `json:"pageProps"`
    } `json:"props"`
}

// NonceResponse 定义了获取 nonce 的响应结构
type NonceResponse struct {
    Uuid string
}

// UploadResponse 定义了文件上传的响应结构
type UploadResponse struct {
    Filename     string `json:"filename"`
    UserFilename string `json:"user_filename"`
}

// 定义最大查询长度
const MaxQueryLength = 2000

func init() {
    reverseModelMap = getReverseModelMap() // 初始化反向映射

    // 尝试动态获取模型列表
    youModelMap, dynamicList, err := fetchModelList("") // fetchModelList 现在返回 You.com 模型 map
    if err == nil && len(youModelMap) > 0 && len(dynamicList) > 0 {
        dynamicModelMap = youModelMap // 动态模型映射使用 You.com 模型名称 -> You.com 模型名称 (自身)
        dynamicModels = dynamicList
        useDynamicModels = true
        fmt.Println("Successfully fetched dynamic model list from you.com")
    } else {
        fmt.Println("Failed to fetch dynamic model list, using default model map.", err)
        // Fallback to static models if dynamic fetch fails.
        staticModels := make([]ModelDetail, 0, len(modelMap))
        created := time.Now().Unix()
        for openAIModelName := range modelMap { // 遍历 OpenAI 模型名称 (静态 modelMap 的键)
            staticModels = append(staticModels, ModelDetail{
                ID:      openAIModelName, //  ModelDetail ID 存储 OpenAI 模型名称 (用于 /v1/models 响应)
                Object:  "model",
                Created: created,
                OwnedBy: "you.com",
            })
        }
        dynamicModels = staticModels
    }
}

// Handler 是处理所有传入 HTTP 请求的主处理函数。
func Handler(w http.ResponseWriter, r *http.Request) {
    if r.URL.Path == "/v1/models" || r.URL.Path == "/api/v1/models" {
        handleModelsRequest(w, r)
        return
    }
    if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/none/v1/chat/completions" && r.URL.Path != "/such/chat/completions" {
        handleStatusCheck(w, r)
        return
    }
    handleChatCompletions(w, r)
}

func handleModelsRequest(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "*")
    if r.Method == "OPTIONS" {
        w.WriteHeader(http.StatusOK)
        return
    }

    response := ModelResponse{
        Object: "list",
        Data:   dynamicModels, // dynamicModels 的 ModelDetail.ID 已经存储了 OpenAI 模型名称
    }
    json.NewEncoder(w).Encode(response)
}

func handleStatusCheck(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "status":  "You2Api Service Running...",
        "message": "MoLoveSze...",
    })
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
    // 设置 CORS 头部
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "*")
    if r.Method == "OPTIONS" {
        w.WriteHeader(http.StatusOK)
        return
    }

    // 验证 Authorization 头部 
    authHeader := r.Header.Get("Authorization")
    if !strings.HasPrefix(authHeader, "Bearer ") {
        http.Error(w, "Missing or invalid authorization header", http.StatusUnauthorized)
        return
    }
    dsToken := strings.TrimPrefix(authHeader, "Bearer ") // 提取 DS token

    // 解析 OpenAI 请求体
    var openAIReq OpenAIRequest
    if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    originalModel = openAIReq.Model // 保存原始的 OpenAI 模型名称

    // 映射 OpenAI 模型名称到 You.com 模型名称 (request conversion)
    youModelName := mapModelName(openAIReq.Model)
    openAIReq.Model = youModelName // **重要:  替换请求体中的模型名称为 You.com 格式**

    // 转换 system 消息为 user 消息
    openAIReq.Messages = convertSystemToUser(openAIReq.Messages)

    // 构建 You.com 聊天历史
    var chatHistory []map[string]interface{}
    var sources []map[string]interface{}
    var lastAssistantMessage string

    // 处理历史消息（不包括最后一条）
    for _, msg := range openAIReq.Messages[:len(openAIReq.Messages)-1] {
        if msg.Role == "user" {
            tokens, err := countTokens([]Message{msg})
            if err != nil {
                http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
                return
            }
            if tokens > MaxContextTokens {
                // 获取 nonce
                nonceResp, err := getNonce(dsToken)
                if err != nil {
                    fmt.Printf("获取 nonce 失败: %v\n", err)
                    http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
                    return
                }

                // 创建临时文件
                tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
                if err := os.WriteFile(tempFile, []byte(msg.Content), 0644); err != nil {
                    fmt.Printf("创建临时文件失败: %v\n", err)
                    http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
                    return
                }
                defer os.Remove(tempFile)

                // 上传文件
                uploadResp, err := uploadFile(dsToken, tempFile)
                if err != nil {
                    fmt.Printf("上传文件失败: %v\n", err)
                    http.Error(w, "Failed to upload file", http.StatusInternalServerError)
                    return
                }

                // 添加文件源信息
                sources = append(sources, map[string]interface{}{
                    "source_type":   "user_file",
                    "filename":      uploadResp.Filename,
                    "user_filename": uploadResp.UserFilename,
                    "size_bytes":    len(msg.Content),
                })

                // 在历史记录中使用文件引用
                chatHistory = append(chatHistory, map[string]interface{}{
                    "question": fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename),
                    "answer":   "",
                })
            } else {
                chatHistory = append(chatHistory, map[string]interface{}{
                    "question": msg.Content,
                    "answer":   "",
                })
            }
        } else if msg.Role == "assistant" {
            // 只保存最后一条 assistant 消息 
            lastAssistantMessage = msg.Content
        }
    }

    // 如果有最后一条 assistant 消息，添加到历史记录中
    if lastAssistantMessage != "" {
        chatHistory = append(chatHistory, map[string]interface{}{
            "question": "",
            "answer":   lastAssistantMessage,
        })
    }
    chatHistoryJSON, _ := json.Marshal(chatHistory)

    // 创建 You.com API 请求
    youReq, _ := http.NewRequest("GET", "https://you.com/api/streamingSearch", nil)

    // 生成必要的 ID
    chatId := uuid.New().String()
    conversationTurnId := uuid.New().String()
    traceId := fmt.Sprintf("%s|%s|%s", chatId, conversationTurnId, time.Now().Format(time.RFC3339))

    // 处理最后一条消息
    lastMessage := openAIReq.Messages[len(openAIReq.Messages)-1]
    lastMessageTokens, err := countTokens([]Message{lastMessage})
    if err != nil {
        http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
        return
    }

    // 构建查询参数 (You.com API request)
    q := youReq.URL.Query()

    // 设置基本参数
    q.Add("page", "1")
    q.Add("count", "10")
    q.Add("safeSearch", "Off")
    q.Add("mkt", "en-US")
    q.Add("enable_worklow_generation_ux", "true")
    q.Add("domain", "youchat")
    q.Add("use_personalization_extraction", "true")
    q.Add("queryTraceId", chatId)
    q.Add("chatId", chatId)
    q.Add("conversationTurnId", conversationTurnId)
    q.Add("pastChatLength", fmt.Sprintf("%d", len(chatHistory)))
    q.Add("selectedChatMode", "custom")
    q.Add("selectedAiModel", openAIReq.Model) // **使用已转换为 You.com 格式的模型名称**
    q.Add("enable_agent_clarification_questions", "true")
    q.Add("traceId", traceId)
    q.Add("use_nested_youchat_updates", "true")

    // 如果最后一条消息超过限制，使用文件上传
    if lastMessageTokens > MaxContextTokens {
        // 获取 nonce
        nonceResp, err := getNonce(dsToken)
        if err != nil {
            fmt.Printf("获取 nonce 失败: %v\n", err)
            http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
            return
        }

        // 创建临时文件
        tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
        if err := os.WriteFile(tempFile, []byte(lastMessage.Content), 0644); err != nil {
            fmt.Printf("创建临时文件失败: %v\n", err)
            http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
            return
        }
        defer os.Remove(tempFile)

        // 上传文件
        uploadResp, err := uploadFile(dsToken, tempFile)
        if err != nil {
            fmt.Printf("上传文件失败: %v\n", err)
            http.Error(w, "Failed to upload file", http.StatusInternalServerError)
            return
        }

        // 添加文件源信息
        sources = append(sources, map[string]interface{}{
            "source_type":   "user_file",
            "filename":      uploadResp.Filename,
            "user_filename": uploadResp.UserFilename,
            "size_bytes":    len(lastMessage.Content),
        })

        // 添加 sources 参数
        sourcesJSON, _ := json.Marshal(sources)
        q.Add("sources", string(sourcesJSON))

        // 使用文件引用作为查询
        q.Add("q", fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename))
    } else {
        // 如果有之前上传的文件，添加 sources
        if len(sources) > 0 {
            sourcesJSON, _ := json.Marshal(sources)
            q.Add("sources", string(sourcesJSON))
        }
        q.Add("q", lastMessage.Content)
    }
    q.Add("chat", string(chatHistoryJSON))
    youReq.URL.RawQuery = q.Encode()

    fmt.Printf("\n=== 完整请求信息 ===\n")
    fmt.Printf("请求 URL: %s\n", youReq.URL.String())
    fmt.Printf("请求头:\n")
    for key, values := range youReq.Header {
        fmt.Printf("%s: %v\n", key, values)
    }

    // 设置请求头
    youReq.Header = http.Header{
        "sec-ch-ua-platform":         {"Windows"},
        "Cache-Control":              {"no-cache"},
        "sec-ch-ua":                  {`"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"`},
        "sec-ch-ua-bitness":          {"64"},
        "sec-ch-ua-model":            {""},
        "sec-ch-ua-mobile":           {"?0"},
        "sec-ch-ua-arch":             {"x86"},
        "sec-ch-ua-full-version":     {"133.0.3065.39"},
        "Accept":                     {"text/event-stream"},
        "User-Agent":                 {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"},
        "sec-ch-ua-platform-version": {"19.0.0"},
        "Sec-Fetch-Site":             {"same-origin"},
        "Sec-Fetch-Mode":             {"cors"},
        "Sec-Fetch-Dest":             {"empty"},
        "Host":                       {"you.com"},
    }

    // 设置 Cookie
    cookies := getCookies(dsToken)
    var cookieStrings []string
    for name, value := range cookies {
        cookieStrings = append(cookieStrings, fmt.Sprintf("%s=%s", name, value))
    }
    youReq.Header.Add("Cookie", strings.Join(cookieStrings, ";"))
    fmt.Printf("Cookie: %s\n", strings.Join(cookieStrings, ";"))
    fmt.Printf("===================\n\n")

    // 发送请求并获取响应
    client := &http.Client{}
    resp, err := client.Do(youReq)
    if err != nil {
        fmt.Printf("发送请求失败: %v\n", err)
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer resp.Body.Close()

    // 打印响应状态码
    fmt.Printf("响应状态码: %d\n", resp.StatusCode)

    // 如果状态码不是 200，打印响应内容
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        fmt.Printf("错误响应内容: %s\n", string(body))
        http.Error(w, fmt.Sprintf("API returned status %d", resp.StatusCode), resp.StatusCode)
        return
    }

    // 根据 OpenAI 请求的 stream 参数选择处理函数
    if !openAIReq.Stream {
        handleNonStreamingResponse(w, youReq, originalModel)
        return
    }
    handleStreamingResponse(w, youReq, originalModel)
}

// getCookies 根据提供的 DS token 生成所需的 Cookie。
func getCookies(dsToken string) map[string]string {
    return map[string]string{
        "guest_has_seen_legal_disclaimer": "true",
        "youchat_personalization":         "true",
        "DS":                              dsToken, // 关键的 DS token
        "you_subscription":                "youpro_standard_year", // 示例订阅信息
        "youpro_subscription":             "true",
        "ai_model":                        "deepseek_r1",         // 示例 AI 模型
        "youchat_smart_learn":             "true",
    }
}

// handleNonStreamingResponse 处理非流式请求。
func handleNonStreamingResponse(w http.ResponseWriter, youReq *http.Request, originalModelName string) {
    client := &http.Client{
        Timeout: 60 * time.Second, // 设置超时时间
    }
    resp, err := client.Do(youReq)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer resp.Body.Close()

    var fullResponse strings.Builder
    scanner := bufio.NewScanner(resp.Body)
    // 设置 scanner 的缓冲区大小（可选，但对于大型响应很重要）
    buf := make([]byte, 0, 64*1024)
    scanner.Buffer(buf, 1024*1024)

    // 逐行扫描响应，寻找 youChatToken 事件
    for scanner.Scan() {
        line := scanner.Text()
        if strings.HasPrefix(line, "event: youChatToken") {
            scanner.Scan() // 读取下一行 (data 行)
            data := scanner.Text()
            if !strings.HasPrefix(data, "data: ") {
                continue // 如果不是 data 行，则跳过
            }
            var token YouChatResponse
            if err := json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token); err != nil {
                continue // 如果解析失败，则跳过
            }
            fullResponse.WriteString(token.YouChatToken) // 将 token 添加到完整响应中
        }
    }

    if scanner.Err() != nil {
        http.Error(w, "Error reading response", http.StatusInternalServerError)
        return
    }

    // 构建 OpenAI 格式的非流式响应
    openAIResp := OpenAIResponse{
        ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   originalModelName, // 使用 originalModelName (OpenAI 风格)
        Choices: []OpenAIChoice{
            {
                Message: Message{
                    Role:    "assistant",
                    Content: fullResponse.String(), // 完整的响应内容
                },
                Index:        0,
                FinishReason: "stop", // 停止原因
            },
        },
    }

    w.Header().Set("Content-Type", "application/json")
    if err := json.NewEncoder(w).Encode(openAIResp); err != nil {
        http.Error(w, "Error encoding response", http.StatusInternalServerError)
        return
    }
}

// handleStreamingResponse 处理流式请求。
func handleStreamingResponse(w http.ResponseWriter, youReq *http.Request, originalModelName string) {
    client := &http.Client{} // 流式请求不需要设置超时，因为它会持续接收数据
    resp, err := client.Do(youReq)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer resp.Body.Close()

    // 设置流式响应的头部
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    scanner := bufio.NewScanner(resp.Body)
    // 逐行扫描响应，寻找 youChatToken 事件
    for scanner.Scan() {
        line := scanner.Text()
        if strings.HasPrefix(line, "event: youChatToken") {
            scanner.Scan()         // 读取下一行 (data 行)
            data := scanner.Text() // 获取数据行
            var token YouChatResponse
            json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token) // 解析 JSON

            // 构建 OpenAI 格式的流式响应块
            openAIResp := OpenAIStreamResponse{
                ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
                Object:  "chat.completion.chunk",
                Created: time.Now().Unix(),
                Model:   originalModelName, // 使用 originalModelName (OpenAI 风格)
                Choices: []Choice{
                    {
                        Delta: Delta{
                            Content: token.YouChatToken, // 增量内容
                        },
                        Index:        0,
                        FinishReason: "", // 流式响应中通常为空
                    },
                },
            }
            respBytes, _ := json.Marshal(openAIResp)          // 将响应块序列化为 JSON
            fmt.Fprintf(w, "data: %s\n\n", string(respBytes)) // 写入响应数据
            w.(http.Flusher).Flush()                          // 立即刷新输出
        }
    }
}

// 获取上传文件所需的 nonce
func getNonce(dsToken string) (*NonceResponse, error) {
    req, _ := http.NewRequest("GET", "https://you.com/api/get_nonce", nil)
    req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    // 读取完整的响应内容
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("读取响应失败: %v", err)
    }

    // 直接使用响应内容作为 UUID
    return &NonceResponse{
        Uuid: strings.TrimSpace(string(body)),
    }, nil
}

// 上传文件
func uploadFile(dsToken, filePath string) (*UploadResponse, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)
    part, err := writer.CreateFormFile("file", filepath.Base(filePath))
    if err != nil {
        return nil, err
    }
    if _, err := io.Copy(part, file); err != nil {
        return nil, err
    }
    writer.Close()

    req, _ := http.NewRequest("POST", "https://you.com/api/upload", body)
    req.Header.Set("Content-Type", writer.FormDataContentType())
    req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var uploadResp UploadResponse
    if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
        return nil, err
    }
    return &uploadResp, nil
}

// 计算消息的 token 数（使用字符估算方法）
func countTokens(messages []Message) (int, error) {
    totalTokens := 0
    for _, msg := range messages {
        content := msg.Content
        englishCount := 0
        chineseCount := 0

        // 遍历每个字符
        for _, r := range content {
            if r <= 127 { // ASCII 字符（英文和符号）
                englishCount++
            } else { // 非 ASCII 字符（中文等）
                chineseCount++
            }
        }

        // 计算 tokens：英文字符 _0.3 + 中文字符_ 0.6
        tokens := int(float64(englishCount)*0.3 + float64(chineseCount)*1)

        // 加上角色名的 token（约 2 个）
        totalTokens += tokens + 2
    }
    return totalTokens, nil
}

// 将 system 消息转换为第一条 user 消息
func convertSystemToUser(messages []Message) []Message {
    if len(messages) == 0 {
        return messages
    }

    var systemContent strings.Builder
    var newMessages []Message
    var systemFound bool

    // 收集所有 system 消息 
    for _, msg := range messages {
        if msg.Role == "system" {
            if systemContent.Len() > 0 {
                systemContent.WriteString("\n")
            }
            systemContent.WriteString(msg.Content)
            systemFound = true
        } else {
            newMessages = append(newMessages, msg)
        }
    }

    // 如果有 system 消息，将其作为第一条 user 消息
    if systemFound {
        newMessages = append([]Message{{
            Role:    "user",
            Content: systemContent.String(),
        }}, newMessages...)
    }

    return newMessages
}

// fetchModelList 从 you.com 动态获取模型列表
func fetchModelList(dsToken string) (map[string]string, []ModelDetail, error) {
    youReq, _ := http.NewRequest("GET", "https://you.com", nil) // Or the correct page URL
    if dsToken != "" {
        cookies := getCookies(dsToken)
        var cookieStrings []string
        for name, value := range cookies {
            cookieStrings = append(cookieStrings, fmt.Sprintf("%s=%s", name, value))
        }
        youReq.Header.Add("Cookie", strings.Join(cookieStrings, ";"))
    }
    youReq.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0")

    client := &http.Client{}
    resp, err := client.Do(youReq)
    if err != nil {
        return nil, nil, fmt.Errorf("failed to fetch you.com page: %w", err)
    }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, nil, fmt.Errorf("failed to fetch you.com page, status: %d, body: %s", resp.StatusCode, string(body))
    }

    doc, err := goquery.NewDocumentFromReader(resp.Body)
    if err != nil {
        return nil, nil, fmt.Errorf("failed to parse HTML: %w", err)
    }

    script := doc.Find("script#__NEXT_DATA__").Text()
    var nextData NextData
    if err := json.Unmarshal([]byte(script), &nextData); err != nil {
        return nil, nil, fmt.Errorf("failed to parse JSON: %w", err)
    }

    youModelMap := make(map[string]string) // You.com 模型名称 -> You.com 模型名称 (自身映射)
    models := make([]ModelDetail, 0)
    created := time.Now().Unix()
    for _, model := range nextData.Props.PageProps.AiModels {
        youModelID := model.ID
        openaiModelID := reverseMapYouModelNameToOpenAI(youModelID) // 转换为 OpenAI 风格

        youModelMap[openaiModelID] = youModelID //  OpenAI 模型名称 -> You.com 模型名称 (用于动态映射)

        models = append(models, ModelDetail{
            ID:      openaiModelID, // **ModelDetail ID 存储 OpenAI 模型名称 (用于 /v1/models 响应)**
            Object:  "model",
            Created: created,
            OwnedBy: model.Company,
        })
    }

    dynamicModelMap = youModelMap // 更新动态模型映射为 OpenAI -> You.com
    return youModelMap, models, nil // 返回 You.com 模型名称的映射 (实际是 OpenAI -> You.com 映射) 和 ModelDetail 列表
}

// reverseMapYouModelNameToOpenAI 将 You.com 模型 ID 转换为 OpenAI 风格的模型名称 (显式映射 + 正则)
func reverseMapYouModelNameToOpenAI(youModelID string) string {
    // 1. Replace underscores with hyphens (general case, done first)
    replacedHyphens := strings.ReplaceAll(youModelID, "_", "-")

    // 2. Handle double underscore with number in between using regex
    re := regexp.MustCompile(`-(\d+)-`) // Regex to find -digit- pattern (hyphen-digit-hyphen after step 1)
    replacedDecimal := re.ReplaceAllStringFunc(replacedHyphens, func(match string) string {
        // match will be like "-2-"
        numberStr := match[1 : len(match)-1] // Extract the digit part "2"
        return "-" + numberStr + ".0-"       // Replace with "-2.0-" 
    })

    // 3. Dots between digits: No explicit replacement needed because requirement is to *keep* dots.
    //    If there were dots between digits in the original `youModelID` (e.g., "gemini_2.5_pro"), 
    //    and if you wanted to *preserve* them, string.ReplaceAll with "." would do nothing, which is desired.
    

// 显式映射 You.com 模型 ID 到 OpenAI 模型名称
    explicitMap := map[string]string{
        "openai_o3_mini_high":      "o3-mini-high",
        "openai_o3_mini_medium":    "o3-mini-medium",
        "openai_o1":                "o1",
        "openai_o1_preview":        "o1-preview",
        "openai_o1_mini":           "o1-mini",
        "gpt_4o":                   "gpt-4o",
        "gpt_4o_mini":              "gpt-4o-mini",
        "gpt_4_turbo":              "gpt-4-turbo",
        "gpt_3_5":                  "gpt-3.5-turbo",
        "claude_3_opus":            "claude-3-opus",
        "claude_3_sonnet":          "claude-3-sonnet",
        "claude_3_5_sonnet":        "claude-3.5-sonnet",
        "claude_3_5_haiku":         "claude-3.5-haiku",
        "gemini_1_5_pro":           "gemini-1.5-pro",
        "gemini_1_5_flash":         "gemini-1.5-flash",
        "llama3_2_90b":             "llama-3.2-90b",
        "llama3_1_405b":            "llama-3.1-405b",
        "mistral_large_2":          "mistral-large-2",
        "qwen2p5_72b":               "qwen-2.5-72b",
        "qwen2p5_coder_32b":         "qwen-2.5-coder-32b",
        "command_r_plus":             "command-r-plus",
        "claude_3_7_sonnet":          "claude-3.7-sonnet",
        "claude_3_7_sonnet_thinking": "claude-3.7-sonnet-thinking",
        "grok_2":                      "grok-2",          
        "gemini_2_flash":              "gemini-2.0-flash",     
        "deepseek_r1":                "deepseek-r1",
        "deepseek_v3":                "deepseek-chat",
        "mixtral_8x22b":              "mixtral-8x22b",
        "mistral_medium_2402":        "mistral-medium-2402",
        "mistral_small_2402":         "mistral-small-2402",
    }
    if openaiModelName, exists := explicitMap[replacedDecimal]; exists {
        return openaiModelName
    }

    return replacedDecimal
}

    

// getReverseModelMap 创建并返回 You.com 模型名称 -> OpenAI 模型名称 的反向映射。
func getReverseModelMap() map[string]string {
    reverse := make(map[string]string, len(modelMap))
    for openaiName, youName := range modelMap { // 注意这里反转了键值
        reverse[youName] = openaiName
    }
    return reverse
}

// mapModelName 将 OpenAI 模型名称映射到 You.com 模型名称。 (OpenAI -> You.com)
func mapModelName(openAIModel string) string {
    if useDynamicModels {
        if mappedModel, exists := dynamicModelMap[openAIModel]; exists { // dynamicModelMap 现在是 OpenAI -> You.com
            return mappedModel
        }
    } else {
        if mappedModel, exists := modelMap[openAIModel]; exists { // 静态 modelMap 也是 OpenAI -> You.com
            return mappedModel
        }
    }
    return "deepseek_v3" // 默认模型 (You.com name)
}

// reverseMapModelName 将 You.com 模型名称映射回 OpenAI 模型名称。 (You.com -> OpenAI)
func reverseMapModelName(youModel string) string {
    if mappedModel, exists := reverseModelMap[youModel]; exists { // reverseModelMap 是 You.com -> OpenAI
        return mappedModel
    }
    return "deepseek-chat" // 默认模型 (OpenAI name - although default is not critical for reverse mapping)
}

// originalModel 存储原始的 OpenAI 模型名称。 (unchanged)
var originalModel string
