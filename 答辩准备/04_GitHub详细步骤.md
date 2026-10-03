# GitHub 详细步骤（零基础版 · 照着点就行）

> 目标：把 `~/Desktop/电气club作业/` 这个文件夹连它的**8 次修改历史**一起，搬到 github.com 上去，
> 老师点开链接就能看到代码、图、结果表和报告 PDF。
> 预计耗时：**注册 5 分钟 + 建仓 2 分钟 + 推送 1 分钟 = 约 8 分钟**。
> 全程**不用装软件**（PyCharm 自带 git，版本 2.50.1）。

---

## 0. 先搞懂三件事（不然你只是在抄命令，老师一问就露馅）

**① git 和 GitHub 不是一个东西**
- **git** 是你电脑上的一个"版本记录器"。它已经在工作了——你这 8 次修改它全记得（`~/Desktop/电气club作业/.git` 这个隐藏文件夹里）。
- **GitHub** 是个网站，相当于把这个记录**备份到网上**，顺便给别人看。
- 所以：代码其实**早就在你电脑里存好了**，推不推 GitHub 都不会丢。推上去只是为了让别人（老师）能看见。

**② 为什么现在推上去只有一个文件变动？**
git 推的是"改了什么"，不是"整个文件夹重新传一遍"。所以 31 个文件、1.9 MB，几秒钟就传完。

**③ 为什么选 Public（公开）？**
Private 仓库只有你自己能打开，老师点链接会显示 404。招新作业要给人看，必须 Public。
（不用担心"公开了会不会被人抄"——反过来想，公开仓库本身就是你的作品证明，有提交时间可查。）

---

## 1. 开始前先自检（4 条命令，看清楚自己站在哪）

在 PyCharm 底部点 **Terminal**（没有的话：顶部菜单 View → Tool Windows → Terminal），逐条粘贴、回车：

```bash
git status
git log --oneline | head -5
git remote -v
git ls-files | wc -l
```

**你应该看到的正确结果**（我已经替你跑过了，2026-10-01 18:20 实测）：

| 命令 | 正常结果 | 说明 |
|---|---|---|
| `git status` | `nothing to commit, working tree clean` | 没有没保存的改动，干净 |
| `git log` | 8 行，第一行 `ddfd8d2 补齐作者署名…` | 8 条历史提交，作者都是「徐敏纯」 |
| `git remote -v` | **什么都没显示** | 说明还没连 GitHub——这就是我们要做的 |
| `git ls-files \| wc -l` | `31` | 31 个文件待推送，共 1.9 MB |

**确认过的两件重要的事：**
- `data/`（35040 行原始数据，3 MB）**不在推送列表里**，`git ls-files | grep -c "^data/"` 返回 `0`。这是对的：原始数据不上传，别人按报告里的 DOI 自己下载。
- 要推的 31 个文件里**包含你的定稿 PDF**，但**不包含**那个 1.6 MB 的材料 zip（`.gitignore` 里忽略了 `*.zip`）。

---

## 2. 注册账号（约 5 分钟）

> ⚠️ **先换手机热点。** 你宿舍/校园网会拦截部分 GitHub 请求（之前 Overleaf 就是这个原因），
> 表现为页面打不开、白屏、按钮点了没反应。换了热点顺畅很多。

### 2.1 打开浏览器（要真的浏览器）

用 **Safari 或 Chrome**，地址栏输 `github.com`。

> ❌ 不要用任何软件里的"内置预览面板"——那种面板跑不了网页脚本，会白屏显示
> `Please enable JS and disable any ad blocker`。看着像断网，其实是面板的问题，别被误导去重装浏览器。
> 判断网络到底通不通：终端里跑 `curl -sS -o /dev/null -w "%{http_code}" https://github.com/`，
> 返回 `200` 就是通的（我刚跑过，通）。

### 2.2 点 Sign up

右上角 **Sign up**（绿色的）。填三样：

| 字段 | 怎么填 | 为什么 |
|---|---|---|
| **Username（用户名）** | 只能英文/数字/短横线，例如 `xuminchun2026` | 这是你仓库地址的前缀，**一旦定了改不了**。建议就用名字拼音+年份 |
| **Email** | `1037899152@qq.com` | 要和报告里的邮箱一致 |
| **Password** | **12 个字符以上** | GitHub 现在最短 12 位，短了不让你注册 |

**把用户名抄在这条横线上，后面三步都要用：** ______________________

### 2.3 去邮箱点激活链接

注册完 GitHub 会发一封邮件到 QQ 邮箱，里面有激活链接，**必须点**。不点后面建仓库会卡住。
（没收到就看一下 QQ 邮箱的垃圾箱。）

### 2.4 万一白屏 / 提示"请求太多次"

这是 GitHub 的**防刷限流**，按网络出口 IP 算的——不是你电脑坏了，也不关浏览器的事。按顺序应对：

1. **立刻停手**，不要再点。每点一次计数 +1，冷却时间越拖越久。
2. 等 **30~60 分钟**（期间去背你的 6 个数字，别浪费时间）。
3. **换网络**：手机切 4G/5G，或者换个 WiFi（等于换 IP，限流计数通常就重置了）。
4. 换完网络**只点一次**，一次把表单填完提交。

---

## 3. 建一个空仓库（约 2 分钟）

登录后：

1. 点右上角 **`+`** → **New repository**
2. **Repository name** 填：`pv-power-forecasting`（只能小写字母、数字、短横线）
3. **Description** 填：`Photovoltaic power forecasting: baseline vs machine learning (Ningxia 2019, 15-min)`（可不填）
4. **Visibility：选 Public** ← 选错了老师打不开
5. ⚠️ **不要勾** "Add a README file"（三个勾都别动）——我们本地已经有 README 了，勾了两边会打架
6. 点绿色 **Create repository**

做完会跳到一个页面，上面显示 "Quick setup" 和几行命令——**那些命令不用管**，我们用自己的。

---

## 4. 把代码推上去

### 路线 A：命令行（3 条，推荐，最快）

PyCharm 底部的 **Terminal** 里，粘贴这三条（把 `你的用户名` 换成第 2.2 步抄下来的那个）：

```bash
git remote add origin https://github.com/你的用户名/pv-power-forecasting.git
git branch -M main
git push -u origin main
```

**每条在干嘛（老师可能问，要懂）：**

| 命令 | 在干什么 |
|---|---|
| `git remote add origin ...` | 告诉本地 git："我有个远端仓库叫 origin，地址是这条链接"。**只是登记地址，不传任何东西** |
| `git branch -M main` | 把分支名从 `master` 改成 `main`。GitHub 现在默认叫 `main`，名字对不上会推失败 |
| `git push -u origin main` | 真正上传。`-u` 是"记住这个对应关系"，**以后你只打 `git push` 两个词就行** |

> 你现在本机分支已经是 `main` 了（`git branch --show-current` 查过），所以第二条其实是空跑，但**别删**，留着无害。

### 路线 B：GitHub Desktop（图形界面，命令行报错就用它）

如果命令行一直报错，走这条，不用记命令：

1. 装 GitHub Desktop：浏览器打开 `desktop.github.com` → Download → 拖进 Applications
2. 打开 → Sign in with browser → 用刚注册的账号登录
3. 菜单 **File → Add Local Repository** → 选 `~/Desktop/电气club作业` → Add Repository
4. 右上角 **Publish repository** → Name 填 `pv-power-forecasting` → 取消勾 "Keep this code private" → Publish

它会自动处理登录和令牌，**不用你去生成令牌**。

---

## 5. 验证成功（1 分钟）

终端里看到一堆 `+ (31)` 的进度，最后出现 `main -> main` 就成功了。然后：

1. 浏览器打开 `https://github.com/你的用户名/pv-power-forecasting`
2. 应该看到这些文件夹和文件：
   - `code/` `figures/` `results/` `overleaf/` `答辩准备/`
   - `README.md` `run_all.py` `requirements.txt` `.gitignore`
   - `光伏功率预测_徐敏纯_3120260807425.pdf`
3. ⚠️ **确认看不到 `data/`** —— 看不到才是对的（原始数据不分发）
4. 点一下 commit 数量（应该显示 `8 Commits`），进去看作者是不是 **徐敏纯**。
   > 这一步是我今天特意替你修的：之前 7 条提交的作者全是占位符「你的名字」，推上去会很难看，已全部重写成你的名字。

**拿到这个链接后**，可以把它发给老师，或者写进 README 顶部。

---

## 6. 以后改了东西怎么再传（记住这三条，够用半年）

```bash
git add .
git commit -m "写一句这次改了什么"
git push
```

第 2 条 `-m` 后面**必须写一句话**——这是给三个月后的你看的，那时候你一定忘了这次改了啥。

比如：`git commit -m "按老师意见补充 LSTM 对比实验"`

---

## 7. 报错速查

| 看到的报错 | 是什么意思 | 怎么办 |
|---|---|---|
| 弹框要密码，输了 QQ 密码不对 | GitHub 2021 年起**不支持密码登录**了 | 去生成令牌（见下），或用路线 B 的 Desktop |
| `support for password authentication was removed` | 同上 | 同上 |
| `src refspec main does not match any` | 漏了 `git branch -M main` | 补跑一次这条 |
| `failed to push some refs` / `non-fast-forward` | 远端有本地没有的文件 | 先跑 `git pull --rebase origin main` 再 `git push` |
| `Permission denied (publickey)` | 地址选错成 SSH 了 | 确认 remote 地址是 `https://` 开头：`git remote -v` |
| 推送卡住不动 / 超时 | 校园网拦了 | 换手机热点重试 |
| 中文文件名显示成 `\345\205\211...` | 终端编码显示问题 | 不影响，网页上看是正常的 |

**生成令牌的路子**（命令行登录用）：
GitHub 网页 → 右上角头像 → **Settings** → 左侧最下面 **Developer settings** →
**Personal access tokens → Tokens (classic)** → **Generate new token (classic)** →
勾 `repo` → 生成 → **复制那一大串**（关掉页面就看不到了）→
推送时用户名填你的 GitHub 用户名，**密码那一栏粘贴这串令牌**。

---

## 8. 时间预算与建议

| 阶段 | 时间 | 备注 |
|---|---|---|
| 换热点 + 注册 + 激活 | 5 min | 卡住就等 30 分钟再来，别硬点 |
| 建空仓库 | 2 min | 记住别勾 README |
| 推送 + 验证 | 2 min | |
| **合计** | **约 10 min** | |

**我的建议**：今晚睡前做。做不动就明天——**它不阻塞答辩**，你的 PDF 里没有 GitHub 链接，报告本身是完整的。
但它是加分项：推上去之后，"这份作品从 9 月 29 号开始、有 8 次可查的修改记录"这件事就成立了，
比口头说"我做了两周"有说服力得多。
