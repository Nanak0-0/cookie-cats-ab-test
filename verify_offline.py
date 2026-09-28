import io, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, 'index.html')
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
WIDTHS = [500, 900, 1400]

PROBE = r'''
<script>
setTimeout(function(){
  var r = {};
  r.echarts = window.echarts ? window.echarts.version : 'MISSING';
  var cvs = document.querySelectorAll('canvas');
  r.canvas = cvs.length; r.blank = 0; r.dims = [];
  for (var i = 0; i < cvs.length; i++) {
    var c = cvs[i], ratio = -1;
    try {
      var d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      var p = 0, t = 0;
      for (var k = 3; k < d.length; k += 148) { t++; if (d[k] > 8) p++; }
      ratio = p / t;
    } catch (e) { ratio = -2; }
    if (ratio >= 0 && ratio < 0.01) r.blank++;
    r.dims.push(c.width + 'x' + c.height + ':' + (ratio < 0 ? 'ERR' : ratio.toFixed(3)));
  }
  r.iw = window.innerWidth; r.sw = document.documentElement.scrollWidth;
  r.overflow = r.sw > r.iw;
  r.charts = document.querySelectorAll('.chart').length;
  r.tables = document.querySelectorAll('table').length;
  r.nav = document.querySelectorAll('.toc a').length;
  r.kpi = document.querySelectorAll('.kpi').length;
  r.bootFail = 0;
  var hs = document.querySelectorAll('.note-card.warn h4');
  for (var j = 0; j < hs.length; j++) { if (hs[j].textContent.indexOf('未能初始化') >= 0) r.bootFail++; }
  document.title = 'PROBEJSON' + JSON.stringify(r) + 'ENDPROBE';
}, 4500);
</script>
'''

CHROME_FLAGS = [
    '--headless=new', '--disable-gpu', '--no-proxy-server',
    '--disable-background-networking', '--disable-component-update',
    '--host-resolver-rules=MAP * ~NOTFOUND', 
    '--virtual-time-budget=30000',
]


def inject(html):
    head, sep, tail = html.rpartition('</body>')
    assert sep, '未找到 </body>'
    return head + PROBE + sep + tail


def run_probe(html, width, tmpdir):
    page = os.path.join(tmpdir, 'probe_%d.html' % width)
    io.open(page, 'w', encoding='utf-8', newline='\n').write(html)
    cmd = [CHROME] + CHROME_FLAGS + ['--window-size=%d,3000' % width, '--dump-dom',
                                     'file:///' + page.replace('\\', '/')]
    out = subprocess.run(cmd, capture_output=True, timeout=180).stdout.decode('utf-8', 'replace')
    m = re.search(r'PROBEJSON\{[^{}]*\}ENDPROBE', out)
    if not m:
        raise RuntimeError('视口 %d 未取得探针结果（页面可能未正确加载）' % width)
    return json.loads(m.group(0)[len('PROBEJSON'):-len('ENDPROBE')])


def main():
    html = io.open(REPORT, encoding='utf-8').read()

    print('=== 静态检查：外部网络依赖 ===')
    problems = []
    if re.search(r'<script[^>]*\ssrc=', html, re.I):
        problems.append('存在外链 <script src>')
    if re.search(r'<link[^>]*\shref=', html, re.I):
        problems.append('存在外链 <link href>')
    for css in re.findall(r'<style[^>]*>(.*?)</style>', html, re.S | re.I):
        if re.search(r'@import', css, re.I):
            problems.append('CSS 存在 @import')
        if re.search(r'url\(\s*["\']?(?:https?:)?//', css, re.I):
            problems.append('CSS 存在 url() 远程引用')
    for tok in ('cdn.jsdelivr', 'unpkg.com', 'bootcdn', 'cdnjs', 'googleapis'):
        if tok in html:
            problems.append('残留 CDN 关键字 ' + tok)
    size = os.path.getsize(REPORT)
    print('  文件大小        : %d 字节 (%.2f MB)' % (size, size / 1048576))
    print('  script 标签数   : %d（应全部内联）' % len(re.findall(r'<script', html, re.I)))
    print('  外链资源问题    : %s' % (problems if problems else '无'))
    if problems:
        return 1

    print()
    print('=== 运行时检查：断网 + 多视口 ===')
    html_probe = inject(html)
    allok = True
    with tempfile.TemporaryDirectory() as tmp:
        for w in WIDTHS:
            r = run_probe(html_probe, w, tmp)
            ok = (r['echarts'] == '5.5.1' and r['canvas'] == r['charts']
                  and r['blank'] == 0 and not r['overflow'] and r['bootFail'] == 0)
            allok = allok and ok
            print('  视口 %-5d %s  echarts=%-6s 图表=%d/%d  空白画布=%d  '
                  '横向溢出=%s  图库失败提示=%d'
                  % (w, 'OK  ' if ok else 'FAIL', r['echarts'], r['canvas'], r['charts'],
                     r['blank'], r['overflow'], r['bootFail']))
            if w == 1400:
                for d in r['dims']:
                    print('        canvas %s' % d)
            print('        KPI/表/导航 = %d / %d / %d'
                  % (r.get('kpi', 0), r['tables'], r['nav']))

    print()
    print('结果：', 'PASS —— 完全离线可用，所有图表渲染正常' if allok else 'FAIL')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
